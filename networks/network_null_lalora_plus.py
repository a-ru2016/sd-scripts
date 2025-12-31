import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
import logging
import numpy as np

# --- QLoRA Support ---
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

logger = logging.getLogger(__name__)

# ==========================================
# 1. Helper Functions (Fixed for Null Space)
# ==========================================

def get_dequantized_weight(module):
    """
    QLoRA対応: 重み取得
    """
    if bnb and hasattr(module, "weight") and hasattr(module.weight, "quant_state"):
        weight = module.weight
        return bnb.functional.dequantize_4bit(weight, weight.quant_state).to(torch.float32)
    elif bnb and module.__class__.__name__ == "Linear8bitLt":
        if hasattr(module, "state") and hasattr(module.state, "SCB"):
             try:
                 return module.weight.detach().float()
             except:
                 return module.weight.float()
        else:
            return module.weight.float()
    elif hasattr(module, "weight"):
        return module.weight
    return None

def compute_null_space_basis(weight, rank, device="cpu"):
    """
    Null-LoRA用: 通常のSVD (Bottom-K) を使用してNull Spaceを抽出
    """
    is_conv = weight.dim() == 4
    if is_conv:
        out_ch, in_ch, k1, k2 = weight.shape
        w_flat = weight.reshape(out_ch, -1)
    else:
        w_flat = weight

    # メモリ不足対策: CPU推奨
    try:
        # Full SVD to get the tail (Null space)
        U, S, Vh = torch.linalg.svd(w_flat.to(device, dtype=torch.float32), full_matrices=False)
    except Exception as e:
        logger.warning(f"SVD failed on device {device}, trying CPU. Error: {e}")
        w_flat_cpu = w_flat.to("cpu", dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(w_flat_cpu, full_matrices=False)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)

    # ランクが元の次元より大きい場合のガード
    actual_rank = min(rank, U.shape[1], Vh.shape[0])
    
    # Null Space = Smallest singular values (Tail)
    u_null = U[:, -actual_rank:].clone()
    v_null = Vh[-actual_rank:, :].clone()

    return u_null, v_null

# ==========================================
# 2. Enhanced Module
# ==========================================

class NullUniversalModule(nn.Module):
    def __init__(
        self, 
        lora_name, 
        org_module, 
        mean_down, basis_down, 
        mean_up, basis_up,
        multiplier=1.0, 
        total_rank=32,
        lalora_lambda=0.0,
        lalora_mean_init=None,
        lalora_precision_init=None,
        loaded_null_down=None,
        loaded_null_up=None,
        loaded_scale=None,
        storage_dtype=torch.float32,
        use_dora=False,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.org_module = org_module
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        
        # --- 修正点: ランクの動的調整 ---
        # 要求されたランクと、ファイルから読み込まれたBasisの実ランクを比較し、
        # 小さい方に合わせることで形状不一致エラーを防ぐ
        requested_r_half = total_rank // 2
        input_rank_down = basis_down.shape[1]
        input_rank_up = basis_up.shape[1]
        
        self.r_half = min(requested_r_half, input_rank_down, input_rank_up)
        if self.r_half < 1: self.r_half = 1
        
        # デバッグログ（ランクが縮小された場合のみ表示）
        if self.r_half < requested_r_half:
            logger.debug(f"[{lora_name}] Rank adjusted: requested {requested_r_half} -> actual {self.r_half} (Basis limit)")

        # Universal Basis (Trainable側生成用)
        # スライスには調整後の self.r_half を使用
        self.register_buffer("mean_down", mean_down.to(storage_dtype))
        self.register_buffer("basis_down", basis_down[:, :self.r_half].to(storage_dtype)) 
        self.register_buffer("mean_up", mean_up.to(storage_dtype))
        self.register_buffer("basis_up", basis_up[:, :self.r_half].to(storage_dtype))     

        # --- Trainable Parameters ---
        # 調整後のランクでAlphaを作成
        self.alpha_down = nn.Parameter(torch.zeros(self.r_half, self.r_half))
        self.alpha_up = nn.Parameter(torch.zeros(self.r_half, self.r_half))
        
        # Architecture Extraction
        if org_module.__class__.__name__ == "Conv2d":
            self.is_conv = True
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.k_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
        elif hasattr(org_module, "in_features") and hasattr(org_module, "out_features"):
            self.is_conv = False
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
        else:
            self.is_conv = False
            w = get_dequantized_weight(org_module)
            if w is not None:
                if w.dim() == 4:
                    self.is_conv = True
                    self.out_dim, self.in_dim, k1, k2 = w.shape
                    self.k_size = (k1, k2)
                    self.stride = (1,1); self.padding = (0,0); self.dilation = (1,1); self.groups = 1
                else:
                    self.out_dim, self.in_dim = w.shape[:2]
            else:
                raise ValueError(f"Could not infer dimensions for module: {lora_name}")

        # Shapes
        if self.is_conv:
             self.down_shape_train = (self.r_half, self.in_dim // self.groups, self.k_size[0], self.k_size[1])
             self.up_shape_train = (self.out_dim, self.r_half, 1, 1)
             self.down_shape_frozen = (self.r_half, self.in_dim // self.groups, self.k_size[0], self.k_size[1])
             self.up_shape_frozen = (self.out_dim, self.r_half, 1, 1)
        else:
             self.down_shape_train = (self.r_half, self.in_dim)
             self.up_shape_train = (self.out_dim, self.r_half)
             self.down_shape_frozen = (self.r_half, self.in_dim)
             self.up_shape_frozen = (self.out_dim, self.r_half)

        # --- DoRA or Standard Scale ---
        if self.use_dora:
            if loaded_scale is not None and loaded_scale.shape[0] == self.out_dim:
                self.m = nn.Parameter(loaded_scale)
            else:
                self.m = nn.Parameter(torch.ones(self.out_dim))
            self.s = None 
        else:
            # 調整後のランクに合わせてScaleパラメータを作成
            if loaded_scale is not None:
                 # ロード時にサイズが合わない場合はトリミングして使用
                 if loaded_scale.shape[0] > self.r_half * 2:
                     self.s = nn.Parameter(loaded_scale[:self.r_half * 2])
                 else:
                     self.s = nn.Parameter(loaded_scale)
            else:
                 self.s = nn.Parameter(torch.ones(self.r_half * 2))

        # --- Null Space Basis (Frozen) ---
        if loaded_null_down is not None and loaded_null_up is not None:
            # Check for legacy swap: null_down (Out, R), null_up (R, In)
            expected_in = self.in_dim if not self.is_conv else ((self.in_dim // self.groups) * self.k_size[0] * self.k_size[1])
            expected_out = self.out_dim
            
            if loaded_null_down.shape[0] == expected_out and loaded_null_up.shape[1] == expected_in:
                 # logger.info(f"[{lora_name}] SVD cache legacy swap detected. Correcting...")
                 loaded_null_down, loaded_null_up = loaded_null_up, loaded_null_down

            # Correct truncation: null_down is (R, In), null_up is (Out, R)
            self.register_buffer("null_down", loaded_null_down[:self.r_half, :].to(storage_dtype)) 
            self.register_buffer("null_up", loaded_null_up[:, :self.r_half].to(storage_dtype))     
        else:
            weight_for_svd = get_dequantized_weight(org_module)
            if weight_for_svd is None:
                 raise ValueError(f"Failed to extract weights from {lora_name} for SVD.")
            
            with torch.no_grad():
                u_null, v_null = compute_null_space_basis(
                    weight_for_svd, 
                    self.r_half, 
                    device=weight_for_svd.device
                )
            
            # v_null is (R, In), u_null is (Out, R)
            self.register_buffer("null_down", v_null.to(storage_dtype)) 
            self.register_buffer("null_up", u_null.to(storage_dtype))   
            
            del weight_for_svd
            torch.cuda.empty_cache()

        # LaLoRA Buffers
        if lalora_mean_init is not None:
            # Init shape match check would be needed, assuming default 0 initialization here for robustness
            self.register_buffer("lalora_mean_down", lalora_mean_init.to(storage_dtype))
            self.register_buffer("lalora_mean_up", lalora_mean_init.to(storage_dtype))
        else:
            self.register_buffer("lalora_mean_down", torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))
            self.register_buffer("lalora_mean_up", torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))

        if lalora_precision_init is not None and lalora_precision_init.dim() > 1:
             self.register_buffer("lalora_precision_down", lalora_precision_init.to(storage_dtype))
             self.register_buffer("lalora_precision_up", lalora_precision_init.to(storage_dtype))
        else:
            self.register_buffer("lalora_precision_down", torch.ones(self.r_half, self.r_half, dtype=storage_dtype))
            self.register_buffer("lalora_precision_up", torch.ones(self.r_half, self.r_half, dtype=storage_dtype))

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

    def get_regularization_loss(self):
        if self.lalora_lambda <= 0.0:
            return torch.tensor(0.0, device=self.alpha_down.device, dtype=self.alpha_down.dtype)
        
        dtype = self.alpha_down.dtype
        diff_d = self.alpha_down - self.lalora_mean_down.to(dtype)
        loss_d = torch.sum((diff_d * diff_d) * self.lalora_precision_down.to(dtype))
        diff_u = self.alpha_up - self.lalora_mean_up.to(dtype)
        loss_u = torch.sum((diff_u * diff_u) * self.lalora_precision_up.to(dtype))
        
        return 0.5 * self.lalora_lambda * (loss_d + loss_u)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        device = x.device
        alpha_d = self.alpha_down.to(device=device, dtype=dtype)
        alpha_u = self.alpha_up.to(device=device, dtype=dtype)
        
        # 1. 重みの生成 (Universal Basis + Alpha)
        basis_d_t = self.basis_down.to(device=device, dtype=dtype).t()
        w_down_generated = (alpha_d @ basis_d_t) + self.mean_down.to(device=device, dtype=dtype).view(1, -1)
        w_up_generated = (self.basis_up.to(device=device, dtype=dtype) @ alpha_u) + self.mean_up.to(device=device, dtype=dtype).view(-1, 1)

        B_train = w_down_generated.view(self.down_shape_train) 
        A_train = w_up_generated.view(self.up_shape_train)     
        
        # Frozen Basis (Null Space)
        if self.is_conv:
            frozen_down = self.null_down.to(device=device, dtype=dtype).view(self.down_shape_frozen) 
            frozen_up = self.null_up.to(device=device, dtype=dtype).view(self.up_shape_frozen)
        else:
            frozen_down = self.null_down.to(device=device, dtype=dtype)
            frozen_up = self.null_up.to(device=device, dtype=dtype)

        # 2. Forward Pass
        if self.use_dora:
            # --- DoRA Logic ---
            if self.is_conv:
                # Path 1: Frozen(down) -> Train(up)
                x1 = F.conv2d(x, frozen_down, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                x1 = F.conv2d(x1, A_train)
                
                # Path 2: Train(down) -> Frozen(up)
                x2 = F.conv2d(x, B_train, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                x2 = F.conv2d(x2, frozen_up)
            else:
                x1 = F.linear(F.linear(x, frozen_down), A_train)
                x2 = F.linear(F.linear(x, B_train), frozen_up)
            
            lx = x1 + x2
            
            # DoRA Magnitude Scaling
            m = self.m.to(device=device, dtype=dtype)
            if self.is_conv:
                lx = lx * m.view(1, -1, 1, 1)
            else:
                lx = lx * m 
                
        else:
            # --- Standard LoRA Logic with Scalar Scale ---
            s = self.s.to(device=device, dtype=dtype)
            s1 = s[:self.r_half] 
            s2 = s[self.r_half:] 
            
            if self.is_conv:
                x1 = F.conv2d(x, frozen_down, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                x1 = x1 * s1.view(1, -1, 1, 1)
                x1 = F.conv2d(x1, A_train)
                
                x2 = F.conv2d(x, B_train, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
                x2 = x2 * s2.view(1, -1, 1, 1)
                x2 = F.conv2d(x2, frozen_up)
            else:
                x1 = F.linear(x, frozen_down)
                x1 = x1 * s1
                x1 = F.linear(x1, A_train)
                
                x2 = F.linear(x, B_train)
                x2 = x2 * s2
                x2 = F.linear(x2, frozen_up)

            lx = x1 + x2


class NullUniversalNetwork(nn.Module):
    def __init__(self, text_encoder, unet, multiplier=1.0, lalora_lambda=0.0, use_dora=False):
        super().__init__()
        self.multiplier = multiplier
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        self.modules_dict = nn.ModuleDict()
        
        self.text_encoder = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.unet = unet

    def load_basis_and_init(self, basis_path, lalora_context_path=None, weights_path=None, svd_cache_path=None, total_rank=32, ignore_te=False, storage_dtype=torch.float32):
        logger.info(f"Loading Universal Basis from: {basis_path}")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis file not found: {basis_path}")

        basis_state_dict = load_file(basis_path)
        
        # Load Context/Weights/Cache
        lalora_context = {}
        if lalora_context_path and os.path.exists(lalora_context_path):
            lalora_context = load_file(lalora_context_path)
            
        loaded_weights = {}
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading weights from: {weights_path}")
            if weights_path.endswith(".safetensors"):
                loaded_weights = load_file(weights_path)
            else:
                loaded_weights = torch.load(weights_path, map_location="cpu")

        loaded_svd_cache = {}
        if svd_cache_path and os.path.exists(svd_cache_path):
            logger.info(f"Loading SVD cache from: {svd_cache_path}")
            loaded_svd_cache = load_file(svd_cache_path)

        # Detect Keys
        keys = set()
        for k in basis_state_dict.keys():
            if k.endswith(".basis"):
                keys.add(k.rsplit(".", 1)[0]) 

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
        logger.info(f"Initializing Null-Universal Modules with Rank={total_rank} (Null-Space SVD Enabled)")
        
        for base_name, parts in layer_groups.items():
            if "down" not in parts or "up" not in parts: continue

            is_te = "te1" in base_name or "te2" in base_name or "text_encoder" in base_name
            if ignore_te and is_te: continue

            target_module = self.find_target_module(base_name)
            
            if target_module is not None:
                mean_d = basis_state_dict[parts["down"] + ".mean"]
                basis_d = basis_state_dict[parts["down"] + ".basis"]
                mean_u = basis_state_dict[parts["up"] + ".mean"]
                basis_u = basis_state_dict[parts["up"] + ".basis"]

                safe_name = base_name.replace(".", "_")
                
                # Weight Loading Logic
                l_mean, l_prec = None, None
                l_null_d, l_null_u, l_scale = None, None, None
                
                def find_key(dct, suffix):
                    k1 = f"{safe_name}.{suffix}"
                    k2 = f"modules_dict.{safe_name}.{suffix}"
                    return dct.get(k1, dct.get(k2, None))

                # Load Buffers
                if loaded_weights:
                    l_null_d = find_key(loaded_weights, "null_down")
                    l_null_u = find_key(loaded_weights, "null_up")
                    if self.use_dora:
                        l_scale = find_key(loaded_weights, "m")
                    else:
                        l_scale = find_key(loaded_weights, "s")
                
                if l_null_d is None and loaded_svd_cache:
                    l_null_d = find_key(loaded_svd_cache, "null_down")
                    l_null_u = find_key(loaded_svd_cache, "null_up")

                u_module = NullUniversalModule(
                    safe_name, target_module,
                    mean_d, basis_d, mean_u, basis_u,
                    multiplier=self.multiplier,
                    total_rank=total_rank,
                    lalora_lambda=self.lalora_lambda,
                    lalora_mean_init=l_mean,
                    lalora_precision_init=l_prec,
                    loaded_null_down=l_null_d,
                    loaded_null_up=l_null_u,
                    loaded_scale=l_scale,
                    storage_dtype=storage_dtype,
                    use_dora=self.use_dora
                )
                
                # Load Trainable Parameters (Alpha)
                if loaded_weights:
                    ad = find_key(loaded_weights, "alpha_down")
                    au = find_key(loaded_weights, "alpha_up")
                    if ad is not None: u_module.alpha_down.data.copy_(ad)
                    if au is not None: u_module.alpha_up.data.copy_(au)
                
                self.modules_dict[safe_name] = u_module
                count += 1
        
        logger.info(f"Initialized {count} Null-Universal modules.")

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
            if len(self.text_encoder) > 0: root = self.text_encoder[0]
            search_key = search_key.replace("lora_te1_", "").replace("te1.", "")
        elif (search_key.startswith("lora_te2_") or search_key.startswith("te2.")) and len(self.text_encoder) > 1:
            root = self.text_encoder[1]
            search_key = search_key.replace("lora_te2_", "").replace("te2.", "")
        
        if root is None: return None
        target_name_converted = search_key.replace(".", "_")
        for name, module in root.named_modules():
            if name.replace(".", "_") == target_name_converted: return module
        return None

    def apply_to(self, text_encoder, unet, train_text_encoder, train_unet):
        for module in self.modules_dict.values():
            module.apply_to()
            
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr, loraplus_lr_ratio=16.0):
        """
        LoRA+ Learning Rate Scheme
        """
        self.requires_grad_(False)
        
        te_params_low = []
        te_params_high = []
        unet_params_low = []
        unet_params_high = []

        safe_default = default_lr if default_lr is not None else 1e-4
        te_lr_base = text_encoder_lr if text_encoder_lr is not None else safe_default
        unet_lr_base = unet_lr if unet_lr is not None else safe_default
        
        # LoRA+ Ratio check
        if loraplus_lr_ratio is None: loraplus_lr_ratio = 1.0

        logger.info(f"LoRA+ Params Setup: Ratio={loraplus_lr_ratio}, TE_LR={te_lr_base}, UNet_LR={unet_lr_base}")

        for name, module in self.modules_dict.items():
            module.alpha_down.requires_grad_(True)
            module.alpha_up.requires_grad_(True)
            
            if self.use_dora:
                module.m.requires_grad_(True)
            else:
                module.s.requires_grad_(True)
            
            is_te = "te1" in name or "te2" in name or "text_encoder" in name
            
            # Group B (low LR): alpha_down, scale
            # Group A (high LR): alpha_up
            if is_te:
                te_params_low.append(module.alpha_down)
                if self.use_dora: te_params_low.append(module.m)
                else: te_params_low.append(module.s)
                
                te_params_high.append(module.alpha_up)
            else:
                unet_params_low.append(module.alpha_down)
                if self.use_dora: unet_params_low.append(module.m)
                else: unet_params_low.append(module.s)
                
                unet_params_high.append(module.alpha_up)

        all_params = []
        
        if te_params_low:
            all_params.append({"params": te_params_low, "lr": te_lr_base})
        if te_params_high:
            all_params.append({"params": te_params_high, "lr": te_lr_base * loraplus_lr_ratio})
            
        if unet_params_low:
            all_params.append({"params": unet_params_low, "lr": unet_lr_base})
        if unet_params_high:
            all_params.append({"params": unet_params_high, "lr": unet_lr_base * loraplus_lr_ratio})

        if self.use_dora:
            return all_params, ["alpha_down", "alpha_up", "m"]
        else:
            return all_params, ["alpha_down", "alpha_up", "s"]

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(False)
        for module in self.modules_dict.values():
            module.alpha_down.requires_grad_(True)
            module.alpha_up.requires_grad_(True)
            if self.use_dora:
                module.m.requires_grad_(True)
            else:
                module.s.requires_grad_(True)

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

    def load_weights(self, file):
        logger.info(f"Loading weights from {file}")
        if file.endswith(".safetensors"):
            weights = load_file(file)
        else:
            weights = torch.load(file, map_location="cpu")

        my_state_dict = {}
        for key, value in weights.items():
            if key.startswith("modules_dict."):
                my_state_dict[key] = value
            else:
                my_state_dict[f"modules_dict.{key}"] = value
        
        info = self.load_state_dict(my_state_dict, strict=False)
        logger.info(f"Weights loaded: {info}")

    def save_weights(self, file, dtype, metadata):
        if metadata is None: metadata = {}
        metadata["lalora_lambda"] = str(self.lalora_lambda)
        metadata["network_dim"] = str(next(iter(self.modules_dict.values())).r_half * 2)
        metadata["lora_plus"] = "true"
        metadata["use_dora"] = str(self.use_dora).lower()
        
        state_dict = {}
        for name, module in self.modules_dict.items():
            state_dict[f"{name}.alpha_down"] = module.alpha_down.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            state_dict[f"{name}.alpha_up"] = module.alpha_up.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            
            if self.use_dora:
                state_dict[f"{name}.m"] = module.m.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            else:
                state_dict[f"{name}.s"] = module.s.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            
            state_dict[f"{name}.null_down"] = module.null_down.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            state_dict[f"{name}.null_up"] = module.null_up.detach().clone().to(dtype if dtype else torch.float16).contiguous()
        
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def save_svd_cache(self, file):
        logger.info(f"Saving SVD cache to {file}...")
        state_dict = {}
        dtype = torch.float32 
        
        for name, module in self.modules_dict.items():
            state_dict[f"{name}.null_down"] = module.null_down.detach().clone().to(dtype).contiguous()
            state_dict[f"{name}.null_up"] = module.null_up.detach().clone().to(dtype).contiguous()
        
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)
        logger.info("SVD cache saved.")

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    lalora_lambda = float(kwargs.get("lalora_lambda", 0.0))
    basis_path = kwargs.get("basis_path", None)
    lalora_context_path = kwargs.get("lalora_context_path", None)
    svd_cache_path = kwargs.get("svd_cache_path", None) 
    use_dora = kwargs.get("use_dora", False)
    if isinstance(use_dora, str):
        use_dora = use_dora.lower() == "true"
    
    network_weights = kwargs.get("network_weights", None)
    if network_weights is None:
        network_weights = kwargs.get("weights", None)

    ignore_te = kwargs.get("ignore_te", None)
    if isinstance(ignore_te, str):
        ignore_te = ignore_te.lower() == "true"
    if ignore_te is None:
        ignore_te = False 

    if basis_path is None: raise ValueError("Null-Universal Network requires 'basis_path'")
    if network_dim is None: network_dim = 32 
    
    network = NullUniversalNetwork(
        text_encoder, unet, 
        multiplier=multiplier, 
        lalora_lambda=lalora_lambda,
        use_dora=use_dora
    )
    
    storage_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            storage_dtype = torch.bfloat16
        else:
            storage_dtype = torch.float16
    
    network.load_basis_and_init(
        basis_path, 
        lalora_context_path, 
        weights_path=network_weights,
        total_rank=network_dim,
        svd_cache_path=svd_cache_path,
        ignore_te=ignore_te,          
        storage_dtype=storage_dtype   
    )
    
    if svd_cache_path is not None and not os.path.exists(svd_cache_path) and network_weights is None:
        network.save_svd_cache(svd_cache_path)
    
    network.text_encoder = None
    network.unet = None
        
    return network

def create_network_from_weights(multiplier, file, vae, text_encoder, unet, **kwargs):
    basis_path = kwargs.get("basis_path", None)
    lalora_lambda = float(kwargs.get("lalora_lambda", 0.0))
    svd_cache_path = kwargs.get("svd_cache_path", None) 
    
    ignore_te = kwargs.get("ignore_te", None)
    if isinstance(ignore_te, str):
         ignore_te = ignore_te.lower() == "true"
    if ignore_te is None: ignore_te = False

    if basis_path is None: raise ValueError("Null-Universal Network requires 'basis_path' even when loading weights")
    
    use_dora = kwargs.get("use_dora", False)
    if isinstance(use_dora, str): use_dora = use_dora.lower() == "true"

    network = NullUniversalNetwork(
        text_encoder, unet, 
        multiplier=multiplier, 
        lalora_lambda=lalora_lambda,
        use_dora=use_dora
    )
    
    storage_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            storage_dtype = torch.bfloat16
        else:
            storage_dtype = torch.float16

    network.load_basis_and_init(
        basis_path, 
        weights_path=file, 
        svd_cache_path=svd_cache_path,
        ignore_te=ignore_te,
        storage_dtype=storage_dtype
    )
    
    network.text_encoder = None
    network.unet = None

    return network, None