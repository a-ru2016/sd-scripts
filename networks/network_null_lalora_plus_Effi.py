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

    try:
        U, S, Vh = torch.linalg.svd(w_flat.to(device, dtype=torch.float32), full_matrices=False)
    except Exception as e:
        logger.warning(f"SVD failed on device {device}, trying CPU. Error: {e}")
        w_flat_cpu = w_flat.to("cpu", dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(w_flat_cpu, full_matrices=False)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)

    actual_rank = min(rank, U.shape[1], Vh.shape[0])
    
    u_null = U[:, -actual_rank:].clone()
    v_null = Vh[-actual_rank:, :].clone()

    return u_null, v_null

# ==========================================
# 2. Enhanced Module with EffiLoRA MoE Support
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
        shared_alpha_down=None, # [EffiLoRA] Shared A Matrix Parameter
        num_experts=1,          # [EffiLoRA] Number of Experts for B Matrix (MoE)
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.org_module = org_module
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        self.num_experts = num_experts
        
        # --- ランクの動的調整 ---
        requested_r_half = total_rank // 2
        input_rank_down = basis_down.shape[1]
        input_rank_up = basis_up.shape[1]
        
        self.r_half = min(requested_r_half, input_rank_down, input_rank_up)
        if self.r_half < 1: self.r_half = 1
        
        if self.r_half < requested_r_half:
            logger.debug(f"[{lora_name}] Rank adjusted: requested {requested_r_half} -> actual {self.r_half} (Basis limit)")

        # Universal Basis
        self.register_buffer("mean_down", mean_down.to(storage_dtype))
        self.register_buffer("basis_down", basis_down[:, :self.r_half].to(storage_dtype)) 
        self.register_buffer("mean_up", mean_up.to(storage_dtype))
        self.register_buffer("basis_up", basis_up[:, :self.r_half].to(storage_dtype))     

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

        # --- Trainable Parameters ---
        
        # 1. Down (A): Shared or Independent
        if shared_alpha_down is not None:
            self.alpha_down = shared_alpha_down
            self.is_alpha_down_shared = True
        else:
            self.alpha_down = nn.Parameter(torch.zeros(self.r_half, self.r_half))
            self.is_alpha_down_shared = False

        # 2. Up (B): MoE Experts
        self.alpha_up = nn.Parameter(torch.zeros(self.num_experts, self.r_half, self.r_half))

        # 3. Router (Gate)
        if self.num_experts > 1:
            self.gate = nn.Linear(self.in_dim, self.num_experts)
            nn.init.normal_(self.gate.weight, std=0.01)
            nn.init.zeros_(self.gate.bias)
            
            # [EffiLoRA] Expert Mask for Reducer
            self.register_buffer("expert_mask", torch.ones(self.num_experts))
        else:
            self.gate = None
            self.register_buffer("expert_mask", torch.ones(1))

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
            if loaded_scale is not None:
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
        
        loss_d = torch.tensor(0.0, device=self.alpha_down.device, dtype=dtype)
        if not self.is_alpha_down_shared:
            diff_d = self.alpha_down - self.lalora_mean_down.to(device=self.alpha_down.device, dtype=dtype)
            loss_d = torch.sum((diff_d * diff_d) * self.lalora_precision_down.to(device=self.alpha_down.device, dtype=dtype))
        
        # MoE Regularization
        diff_u = self.alpha_up - self.lalora_mean_up.to(device=self.alpha_down.device, dtype=dtype).unsqueeze(0)
        loss_u = torch.sum((diff_u * diff_u) * self.lalora_precision_up.to(device=self.alpha_down.device, dtype=dtype).unsqueeze(0))
        
        return 0.5 * self.lalora_lambda * (loss_d + loss_u)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        device = x.device
        # EffiLoRA shared alpha_down might be larger than module's r_half
        alpha_d = self.alpha_down[:self.r_half, :self.r_half].to(device=device, dtype=dtype)
        alpha_u_experts = self.alpha_up.to(device=device, dtype=dtype) # (NumExperts, R, R)
        
        # --- 1. Construct Weights ---
        
        # A Matrix (Down): Shared/Single
        basis_d_t = self.basis_down.to(device=device, dtype=dtype).t()
        w_down_generated = (alpha_d @ basis_d_t) + self.mean_down.to(device=device, dtype=dtype).view(1, -1)
        B_train = w_down_generated.view(self.down_shape_train) 
        
        # Frozen Basis (Null Space)
        if self.is_conv:
            frozen_down = self.null_down.to(device=device, dtype=dtype).view(self.down_shape_frozen) 
            frozen_up = self.null_up.to(device=device, dtype=dtype).view(self.up_shape_frozen)
        else:
            frozen_down = self.null_down.to(device=device, dtype=dtype)
            frozen_up = self.null_up.to(device=device, dtype=dtype)

        # --- 2. Calculate Forward Pass Parts ---
        
        # Part A: Frozen Null Path (Static residue)
        if self.is_conv:
            x_null = F.conv2d(x, frozen_down, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            x_null = F.conv2d(x_null, frozen_up)
        else:
            x_null = F.linear(F.linear(x, frozen_down), frozen_up)

        # Part B: Trainable Path (MoE)
        if self.is_conv:
            x_down = F.conv2d(x, B_train, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            x_down = F.linear(x, B_train)
        
        if self.num_experts > 1:
            # Gate Calculation
            if self.is_conv:
                gate_input = x.mean(dim=(2, 3))
            else:
                if x.dim() == 3:
                     gate_input = x.mean(dim=1)
                else:
                     gate_input = x
            
            gate_logits = F.linear(gate_input.to(device=device, dtype=dtype), self.gate.weight.to(device=device, dtype=dtype), self.gate.bias.to(device=device, dtype=dtype) if self.gate.bias is not None else None) # (B, NumExperts)
            
            # Apply Mask (Reducer)
            # Masked experts should not contribute. We can set their logits to -inf or mask the scores.
            if self.training:
                mask = self.expert_mask.to(device=device, dtype=dtype).view(1, -1) # (1, E)
                # If mask is 0, we want probability to be 0. 
                # So we can multiply probs by mask and re-normalize, or mask logits.
                # Masking logits with large negative value is safer for softmax.
                # But we use simple masking after softmax for gradient control in simple implementation.
                
                gate_scores = F.softmax(gate_logits, dim=-1)
                gate_scores = gate_scores * mask
                # Normalize? If all masked, avoid div by zero.
                sum_scores = gate_scores.sum(dim=-1, keepdim=True) + 1e-6
                gate_scores = gate_scores / sum_scores
            else:
                gate_scores = F.softmax(gate_logits, dim=-1)
            
            # Construct Weighted Experts
            basis_u = self.basis_up.to(device=device, dtype=dtype) 
            mean_u = self.mean_up.to(device=device, dtype=dtype).view(-1, 1)
            
            w_experts = torch.einsum('or, erk -> eok', basis_u, alpha_u_experts) 
            w_experts = w_experts + mean_u.view(1, -1, 1)
            
            x_up_accum = 0
            
            # Apply Experts
            if self.is_conv:
                for i in range(self.num_experts):
                    # Skip if score is negligible (Optimization)
                    if self.training and self.expert_mask[i] == 0: continue
                    
                    w_curr = w_experts[i].view(self.up_shape_train)
                    out_i = F.conv2d(x_down, w_curr)
                    score_i = gate_scores[:, i].view(-1, 1, 1, 1)
                    x_up_accum = x_up_accum + (out_i * score_i)
            else:
                for i in range(self.num_experts):
                    if self.training and self.expert_mask[i] == 0: continue
                    
                    w_curr = w_experts[i] 
                    out_i = F.linear(x_down, w_curr)
                    if out_i.dim() == 3:
                        score_i = gate_scores[:, i].view(-1, 1, 1)
                    else:
                        score_i = gate_scores[:, i].view(-1, 1)
                    x_up_accum = x_up_accum + (out_i * score_i)
            
            x_train = x_up_accum

        else:
            w_up_generated = (self.basis_up.to(device=device, dtype=dtype) @ alpha_u_experts[0]) + self.mean_up.to(device=device, dtype=dtype).view(-1, 1)
            A_train = w_up_generated.view(self.up_shape_train)
            if self.is_conv:
                x_train = F.conv2d(x_down, A_train)
            else:
                x_train = F.linear(x_down, A_train)

        # --- 3. Combine and Scale ---
        if self.use_dora:
            lx = x_null + x_train
            m = self.m.to(device=device, dtype=dtype)
            if self.is_conv:
                lx = lx * m.view(1, -1, 1, 1)
            else:
                lx = lx * m 
        else:
            s = self.s.to(device=device, dtype=dtype)
            x_train = x_train * s[0] 
            lx = x_null + x_train

        return self.org_forward(x) + lx * self.multiplier


class NullUniversalNetwork(nn.Module):
    def __init__(self, text_encoder, unet, multiplier=1.0, lalora_lambda=0.0, use_dora=False, enable_effilora=False, num_experts=4, loraplus_lr_ratio=16.0):
        super().__init__()
        self.multiplier = multiplier
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        self.enable_effilora = enable_effilora
        self.num_experts = num_experts if enable_effilora else 1
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.modules_dict = nn.ModuleDict()
        
        self.text_encoder = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.unet = unet
        
        # [EffiLoRA] Shared Global Parameter
        self.shared_alpha_down = None

    def load_basis_and_init(self, basis_path, lalora_context_path=None, weights_path=None, svd_cache_path=None, total_rank=32, ignore_te=False, storage_dtype=torch.float32):
        logger.info(f"Loading Universal Basis from: {basis_path}")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis file not found: {basis_path}")

        basis_state_dict = load_file(basis_path)
        
        # [EffiLoRA] Init Shared Parameter if enabled
        if self.enable_effilora:
            r_half = total_rank // 2
            self.shared_alpha_down = nn.Parameter(torch.zeros(r_half, r_half))
            logger.info(f"[EffiLoRA] Enabled: Using shared alpha_down matrix (Rank={r_half}) with {self.num_experts} Experts.")

        # Load Weights/Cache
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
                
                l_null_d, l_null_u, l_scale = None, None, None
                
                def find_key(dct, suffix):
                    k1 = f"{safe_name}.{suffix}"
                    k2 = f"modules_dict.{safe_name}.{suffix}"
                    return dct.get(k1, dct.get(k2, None))

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
                    loaded_null_down=l_null_d,
                    loaded_null_up=l_null_u,
                    loaded_scale=l_scale,
                    storage_dtype=storage_dtype,
                    use_dora=self.use_dora,
                    shared_alpha_down=self.shared_alpha_down if self.enable_effilora else None,
                    num_experts=self.num_experts
                )
                
                if loaded_weights:
                    ad = find_key(loaded_weights, "alpha_down")
                    au = find_key(loaded_weights, "alpha_up")
                    gate_w = find_key(loaded_weights, "gate.weight")
                    gate_b = find_key(loaded_weights, "gate.bias")

                    if self.enable_effilora:
                        shared_ad = loaded_weights.get("shared_alpha_down", None)
                        if shared_ad is not None:
                             self.shared_alpha_down.data.copy_(shared_ad)
                        elif ad is not None and count == 0:
                             self.shared_alpha_down.data.copy_(ad)
                    else:
                        if ad is not None: u_module.alpha_down.data.copy_(ad)

                    if au is not None:
                        if au.shape == u_module.alpha_up.shape:
                            u_module.alpha_up.data.copy_(au)
                        elif au.dim() == 2 and u_module.alpha_up.dim() == 3:
                            u_module.alpha_up.data[0].copy_(au)
                    
                    if u_module.gate is not None and gate_w is not None:
                        u_module.gate.weight.data.copy_(gate_w)
                        if gate_b is not None:
                            u_module.gate.bias.data.copy_(gate_b)
                
                self.modules_dict[safe_name] = u_module
                count += 1
        
        logger.info(f"Initialized {count} Null-Universal modules (EffiLoRA={self.enable_effilora}, Experts={self.num_experts}).")

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
            
    # --- [EffiLoRA] Reducer (Dynamic Selection) ---
    def update_effilora_mask(self):
        """
        Refined Reducer: Dynamically select experts based on magnitude (Norm).
        Updates self.expert_mask buffers in each module.
        """
        if not self.enable_effilora or self.num_experts <= 1: return
        
        # We process each module independently for now (Layer-wise suppression)
        # PDF suggests "Layer Suppression" -> Global K layers selection.
        # But here we implement "Expert Suppression" -> K experts per layer.
        
        with torch.no_grad():
            for module in self.modules_dict.values():
                # alpha_up shape: (NumExperts, R, R)
                # Calculate importance (Norm of B matrix)
                norms = module.alpha_up.norm(p=2, dim=(1, 2)) # (NumExperts,)
                
                # Keep top 50% experts active
                k = max(1, int(self.num_experts * 0.5)) 
                
                _, topk_indices = torch.topk(norms, k)
                
                # Update mask
                mask = torch.zeros_like(norms)
                mask[topk_indices] = 1.0
                module.expert_mask.copy_(mask)

    def on_step_start(self, text_encoder, unet):
        # Trigger Reducer at each step
        self.update_effilora_mask()

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr, loraplus_lr_ratio=None):
        self.requires_grad_(False)
        
        te_params_low = []
        te_params_high = []
        unet_params_low = []
        unet_params_high = []
        
        added_shared_down = False
        safe_default = default_lr if default_lr is not None else 1e-4
        te_lr_base = text_encoder_lr if text_encoder_lr is not None else safe_default
        unet_lr_base = unet_lr if unet_lr is not None else safe_default
        
        if loraplus_lr_ratio is None:
            loraplus_lr_ratio = self.loraplus_lr_ratio
        if loraplus_lr_ratio is None: loraplus_lr_ratio = 1.0

        for name, module in self.modules_dict.items():
            if module.is_alpha_down_shared:
                if not added_shared_down:
                    module.alpha_down.requires_grad_(True)
                    unet_params_low.append(module.alpha_down) 
                    added_shared_down = True
            else:
                module.alpha_down.requires_grad_(True)
                if "text_encoder" in name or "te1" in name or "te2" in name:
                    te_params_low.append(module.alpha_down)
                else:
                    unet_params_low.append(module.alpha_down)

            module.alpha_up.requires_grad_(True)
            if module.gate is not None:
                module.gate.requires_grad_(True)

            if self.use_dora:
                module.m.requires_grad_(True)
            else:
                module.s.requires_grad_(True)
            
            is_te = "te1" in name or "te2" in name or "text_encoder" in name
            
            if is_te:
                if self.use_dora: te_params_low.append(module.m)
                else: te_params_low.append(module.s)
                te_params_high.append(module.alpha_up)
                if module.gate is not None: te_params_high.append(module.gate.weight)
                if module.gate is not None and module.gate.bias is not None: te_params_high.append(module.gate.bias)
            else:
                if self.use_dora: unet_params_low.append(module.m)
                else: unet_params_low.append(module.s)
                unet_params_high.append(module.alpha_up)
                if module.gate is not None: unet_params_high.append(module.gate.weight)
                if module.gate is not None and module.gate.bias is not None: unet_params_high.append(module.gate.bias)

        all_params = []
        if te_params_low: all_params.append({"params": te_params_low, "lr": te_lr_base})
        if te_params_high: all_params.append({"params": te_params_high, "lr": te_lr_base * loraplus_lr_ratio})
        if unet_params_low: all_params.append({"params": unet_params_low, "lr": unet_lr_base})
        if unet_params_high: all_params.append({"params": unet_params_high, "lr": unet_lr_base * loraplus_lr_ratio})

        return all_params, ["alpha_down", "alpha_up", "s", "m", "gate"]

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(False)
        for module in self.modules_dict.values():
            module.alpha_down.requires_grad_(True)
            module.alpha_up.requires_grad_(True)
            if module.gate is not None: module.gate.requires_grad_(True)
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

    def save_weights(self, file, dtype, metadata):
        if metadata is None: metadata = {}
        metadata["lalora_lambda"] = str(self.lalora_lambda)
        metadata["network_dim"] = str(next(iter(self.modules_dict.values())).r_half * 2)
        metadata["lora_plus"] = "true"
        metadata["use_dora"] = str(self.use_dora).lower()
        metadata["effilora"] = str(self.enable_effilora).lower()
        metadata["num_experts"] = str(self.num_experts)
        metadata["loraplus_lr_ratio"] = str(self.loraplus_lr_ratio)
        
        state_dict = {}
        
        if self.enable_effilora and self.shared_alpha_down is not None:
            state_dict["shared_alpha_down"] = self.shared_alpha_down.detach().clone().to(dtype if dtype else torch.float16).contiguous()

        for name, module in self.modules_dict.items():
            state_dict[f"{name}.alpha_down"] = module.alpha_down.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            state_dict[f"{name}.alpha_up"] = module.alpha_up.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            
            if module.gate is not None:
                state_dict[f"{name}.gate.weight"] = module.gate.weight.detach().clone().to(dtype if dtype else torch.float16).contiguous()
                if module.gate.bias is not None:
                    state_dict[f"{name}.gate.bias"] = module.gate.bias.detach().clone().to(dtype if dtype else torch.float16).contiguous()

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
        
    enable_effilora = kwargs.get("enable_effilora", False)
    if isinstance(enable_effilora, str):
        enable_effilora = enable_effilora.lower() == "true"
    
    num_experts = int(kwargs.get("num_experts", 4))

    loraplus_lr_ratio = float(kwargs.get("loraplus_lr_ratio", 16.0))

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
        use_dora=use_dora,
        enable_effilora=enable_effilora,
        num_experts=num_experts,
        loraplus_lr_ratio=loraplus_lr_ratio
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
    
    # Do not set text_encoder/unet to None if we need them in on_step_start (though on_step_start receives them as args)
    # network.text_encoder = None
    # network.unet = None
    # Actually, on_step_start in train_network.py is called as: accelerator.unwrap_model(network).on_step_start(text_encoder, unet)
    # So we don't strictly need to keep them in self. But keeping them is safer if we change logic.
    # For memory efficiency, let's follow the standard pattern and release them, 
    # relying on the arguments passed to on_step_start.
    
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
    
    enable_effilora = kwargs.get("enable_effilora", False)
    if isinstance(enable_effilora, str): enable_effilora = enable_effilora.lower() == "true"
    
    num_experts = int(kwargs.get("num_experts", 4))
    loraplus_lr_ratio = float(kwargs.get("loraplus_lr_ratio", 16.0))

    network = NullUniversalNetwork(
        text_encoder, unet, 
        multiplier=multiplier, 
        lalora_lambda=lalora_lambda,
        use_dora=use_dora,
        enable_effilora=enable_effilora,
        num_experts=num_experts,
        loraplus_lr_ratio=loraplus_lr_ratio
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