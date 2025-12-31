import os
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
import logging

logger = logging.getLogger(__name__)

class LaLoRAUniversalModule(nn.Module):
    def __init__(
        self, 
        lora_name, 
        org_module, 
        mean_down, 
        basis_down, 
        mean_up, 
        basis_up, 
        multiplier=1.0, 
        rank=16,
        lalora_lambda=0.0,
        lalora_mean_init=None,
        lalora_precision_init=None
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.org_module = org_module
        self.lalora_lambda = lalora_lambda
        
        # --- Universal Network Buffers (Fixed Basis) ---
        self.register_buffer("mean_down", mean_down)
        self.register_buffer("basis_down", basis_down)
        self.register_buffer("mean_up", mean_up)
        self.register_buffer("basis_up", basis_up)

        # --- Trainable Parameter (Alpha) ---
        self.alpha = nn.Parameter(torch.zeros(rank))

        # --- LaLoRA Regularization Buffers ---
        if lalora_mean_init is not None:
            self.register_buffer("lalora_mean", lalora_mean_init)
        else:
            self.register_buffer("lalora_mean", torch.zeros(rank))

        if lalora_precision_init is not None:
            self.register_buffer("lalora_precision", lalora_precision_init)
        else:
            self.register_buffer("lalora_precision", torch.ones(rank))

        # --- Architecture Detection ---
        if org_module.__class__.__name__ == "Conv2d":
            self.is_conv = True
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
            
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            
            down_len = mean_down.shape[0]
            self.lora_rank = down_len // (in_dim * k_size[0] * k_size[1])
            
            self.down_shape = (self.lora_rank, in_dim, k_size[0], k_size[1])
            self.up_shape = (org_module.out_channels, self.lora_rank, 1, 1)
        else:
            self.is_conv = False
            self.lora_rank = mean_down.shape[0] // org_module.in_features
            self.down_shape = (self.lora_rank, org_module.in_features)
            self.up_shape = (org_module.out_features, self.lora_rank)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

    def get_regularization_loss(self):
        if self.lalora_lambda <= 0.0:
            return torch.tensor(0.0, device=self.alpha.device, dtype=self.alpha.dtype)

        diff = self.alpha - self.lalora_mean
        weighted_sq_diff = (diff * diff) * self.lalora_precision
        loss = 0.5 * self.lalora_lambda * torch.sum(weighted_sq_diff)
        return loss

    # 修正箇所: *args, **kwargs を受け取るように変更
    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        alpha = self.alpha.to(dtype)
        
        # Reconstruct weights
        w_down_flat = self.mean_down.to(dtype) + (self.basis_down.to(dtype) @ alpha)
        w_up_flat = self.mean_up.to(dtype) + (self.basis_up.to(dtype) @ alpha)

        w_down = w_down_flat.view(self.down_shape)
        w_up = w_up_flat.view(self.up_shape)

        # Apply LoRA path
        if self.is_conv:
            lx = torch.nn.functional.conv2d(x, w_down, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            lx = torch.nn.functional.conv2d(lx, w_up)
        else:
            lx = torch.nn.functional.linear(x, w_down)
            lx = torch.nn.functional.linear(lx, w_up)

        # org_forwardには通常 x だけを渡すのが安全です
        # (diffusersのLoRACompatibleLinearなどが元の層だった場合でも、ベースウェイトの適用には x だけで十分なため)
        return self.org_forward(x) + lx * self.multiplier


class UniversalNetwork(nn.Module):
    def __init__(self, text_encoder, unet, multiplier=1.0, lalora_lambda=0.0):
        super().__init__()
        self.multiplier = multiplier
        self.lalora_lambda = lalora_lambda
        self.modules_dict = nn.ModuleDict()
        
        self.text_encoder = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.unet = unet

    def load_basis(self, basis_path, lalora_context_path=None):
        logger.info(f"Loading Universal Basis from: {basis_path}")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis file not found: {basis_path}")

        state_dict = load_file(basis_path)
        
        lalora_context = {}
        if lalora_context_path and os.path.exists(lalora_context_path):
            logger.info(f"Loading LaLoRA context from: {lalora_context_path}")
            lalora_context = load_file(lalora_context_path)
        
        keys = set()
        basis_rank = 0
        for k in state_dict.keys():
            if k.endswith(".basis"):
                if basis_rank == 0:
                    basis_rank = state_dict[k].shape[1]
                keys.add(k.rsplit(".", 1)[0]) 
        
        logger.info(f"Basis Rank detected: {basis_rank}")

        layer_groups = {}
        for k in keys:
            if "lora_down" in k:
                base_name = k.split(".lora_down")[0]
                if base_name not in layer_groups: layer_groups[base_name] = {}
                layer_groups[base_name]["down"] = k
            elif "lora_up" in k:
                base_name = k.split(".lora_up")[0]
                if base_name not in layer_groups: layer_groups[base_name] = {}
                layer_groups[base_name]["up"] = k

        count = 0
        for base_name, parts in layer_groups.items():
            if "down" not in parts or "up" not in parts:
                continue

            target_module = self.find_target_module(base_name)
            
            if target_module is not None:
                mean_d = state_dict[parts["down"] + ".mean"]
                basis_d = state_dict[parts["down"] + ".basis"]
                mean_u = state_dict[parts["up"] + ".mean"]
                basis_u = state_dict[parts["up"] + ".basis"]
                
                if basis_d.shape[1] != basis_rank:
                    continue

                safe_name = base_name.replace(".", "_")
                
                l_mean = None
                l_prec = None
                
                alpha_mean_key = f"{base_name}.alpha_mean"
                alpha_prec_key = f"{base_name}.alpha_precision"
                
                if alpha_mean_key in lalora_context: l_mean = lalora_context[alpha_mean_key]
                elif f"{safe_name}.alpha_mean" in lalora_context: l_mean = lalora_context[f"{safe_name}.alpha_mean"]
                
                if alpha_prec_key in lalora_context: l_prec = lalora_context[alpha_prec_key]
                elif f"{safe_name}.alpha_precision" in lalora_context: l_prec = lalora_context[f"{safe_name}.alpha_precision"]

                u_module = LaLoRAUniversalModule(
                    base_name, target_module,
                    mean_d, basis_d, mean_u, basis_u,
                    multiplier=self.multiplier,
                    rank=basis_rank,
                    lalora_lambda=self.lalora_lambda,
                    lalora_mean_init=l_mean,
                    lalora_precision_init=l_prec
                )
                self.modules_dict[safe_name] = u_module
                count += 1
        
        logger.info(f"Universal Network (LaLoRA ready): Initialized {count} modules. Lambda: {self.lalora_lambda}")

    def find_target_module(self, lora_key):
        search_key = lora_key
        root = None

        if search_key.startswith("lora_unet_"):
            root = self.unet
            search_key = search_key.replace("lora_unet_", "")
        elif search_key.startswith("unet."):
            root = self.unet
            search_key = search_key.replace("unet.", "")
        elif search_key.startswith("lora_te1_") or search_key.startswith("te1."):
            root = self.text_encoder[0]
            search_key = search_key.replace("lora_te1_", "").replace("te1.", "")
        elif (search_key.startswith("lora_te2_") or search_key.startswith("te2.")) and len(self.text_encoder) > 1:
            root = self.text_encoder[1]
            search_key = search_key.replace("lora_te2_", "").replace("te2.", "")
        
        if root is None:
            return None
        
        target_name_converted = search_key.replace(".", "_")
        
        for name, module in root.named_modules():
            normalized_name = name.replace(".", "_")
            if normalized_name == target_name_converted:
                return module
        
        return None

    def apply_to(self, text_encoder, unet, train_text_encoder, train_unet):
        for module in self.modules_dict.values():
            module.apply_to()
            
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(False)
        params = []
        for module in self.modules_dict.values():
            module.alpha.requires_grad_(True)
            params.append(module.alpha)
        return [{"params": params, "lr": default_lr}], ["universal_alpha"]

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(False)
        for module in self.modules_dict.values():
            module.alpha.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def enable_gradient_checkpointing(self):
        pass
        
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for module in self.modules_dict.values():
            module.multiplier = multiplier

    def get_regularization_loss(self):
        total_reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.lalora_lambda <= 0.0:
            return total_reg_loss
            
        for module in self.modules_dict.values():
            total_reg_loss += module.get_regularization_loss()
            
        return total_reg_loss

    def save_weights(self, file, dtype, metadata):
        if metadata is None: metadata = {}
        metadata["lalora_lambda"] = str(self.lalora_lambda)
        
        state_dict = {}
        for name, module in self.modules_dict.items():
            key = f"{module.lora_name}.alpha"
            state_dict[key] = module.alpha.detach().clone().to(dtype if dtype else torch.float16)
        
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

# --- Entry Points ---

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    lalora_lambda = float(kwargs.get("lalora_lambda", 0.0))
    basis_path = kwargs.get("basis_path", None)
    lalora_context_path = kwargs.get("lalora_context_path", None)

    if basis_path is None:
        raise ValueError("Universal Network requires 'basis_path' in --network_args")

    network = UniversalNetwork(text_encoder, unet, multiplier=multiplier, lalora_lambda=lalora_lambda)
    network.load_basis(basis_path, lalora_context_path)
    return network

def create_network_from_weights(multiplier, file, vae, text_encoder, unet, **kwargs):
    basis_path = kwargs.get("basis_path", None)
    lalora_context_path = kwargs.get("lalora_context_path", None)
    lalora_lambda = float(kwargs.get("lalora_lambda", 0.0))

    if basis_path is None:
        raise ValueError("Universal Network requires 'basis_path' in --network_args")
        
    network = UniversalNetwork(text_encoder, unet, multiplier=multiplier, lalora_lambda=lalora_lambda)
    network.load_basis(basis_path, lalora_context_path)
    
    if os.path.exists(file):
        weights = load_file(file)
        for name, module in network.modules_dict.items():
            key = f"{module.lora_name}.alpha"
            if key in weights:
                module.alpha.data.copy_(weights[key])
    return network, None