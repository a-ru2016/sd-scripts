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

    # Null Space tuning usually takes the TAIL of the SVD
    # However, standard LoRA takes the HEAD.
    # Assuming we want the components LEAST represented in the original weight
    # to pivot off of.
    actual_rank = min(rank, U.shape[1], Vh.shape[0])
    
    u_null = U[:, -actual_rank:].clone()
    v_null = Vh[-actual_rank:, :].clone()

    del U, S, Vh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return u_null, v_null

# ==========================================
# 2. Enhanced Module with EffiLoRA MoE & ABM Support
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
        shared_alpha_down=None,
        num_experts=1,
        abm_batch_size=None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.org_module = org_module
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        self.num_experts = num_experts
        
        # ABM Support
        self.abm_enabled = False
        self.abm_margin = 0.5
        self.abm_weight = 1.0
        self.abm_loss = 0.0
        self.abm_batch_size = abm_batch_size
        
        # Rank Splitting (Half Trainable, Half Frozen)
        self.r_half = total_rank // 2
        if self.r_half < 1: self.r_half = 1
        
        # Architecture Extraction (Moved up to cap r_half)
        if isinstance(org_module, nn.Conv2d) or org_module.__class__.__name__ == "Conv2d":
            self.is_conv = True
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.k_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
        elif isinstance(org_module, nn.Linear) or (hasattr(org_module, "in_features") and hasattr(org_module, "out_features")):
            self.is_conv = False
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
            self.k_size = (1, 1)
            self.stride = (1, 1); self.padding = (0, 0); self.dilation = (1, 1); self.groups = 1
        else:
            # Fallback for quantized linear or other module types
            self.is_conv = False
            w = get_dequantized_weight(org_module)
            if w is not None:
                if w.dim() == 4:
                    self.is_conv = True
                    self.out_dim, in_dim_per_group, k1, k2 = w.shape
                    self.groups = getattr(org_module, "groups", 1)
                    self.in_dim = in_dim_per_group * self.groups
                    self.k_size = (k1, k2)
                    self.stride = getattr(org_module, "stride", (1, 1))
                    self.padding = getattr(org_module, "padding", (0, 0))
                    self.dilation = getattr(org_module, "dilation", (1, 1))
                else:
                    self.out_dim, self.in_dim = w.shape[:2]
                    self.groups = 1
                    self.k_size = (1, 1)
                    self.stride = (1, 1); self.padding = (0, 0); self.dilation = (1, 1)
            else:
                raise ValueError(f"Could not infer dimensions for module: {lora_name}")

        # Cap r_half by maximum possible rank
        if self.is_conv:
            max_rank = min(self.out_dim, (self.in_dim // self.groups) * self.k_size[0] * self.k_size[1])
        else:
            max_rank = min(self.out_dim, self.in_dim)
        
        if self.r_half > max_rank:
            self.r_half = max_rank

        # Set frozen shapes
        if self.is_conv:
            self.down_shape_frozen = (self.out_dim, self.r_half, 1, 1)
            self.up_shape_frozen = (self.r_half, self.in_dim // self.groups, self.k_size[0], self.k_size[1])
        else:
            self.down_shape_frozen = (self.out_dim, self.r_half)
            self.up_shape_frozen = (self.r_half, self.in_dim)

        # --- Universal Basis (For Trainable Path) ---
        if mean_down.dim() == 1:
            mean_down_reshaped = mean_down.unsqueeze(0)
        else:
            mean_down_reshaped = mean_down.view(1, -1)
            
        basis_down_t = basis_down[:, :self.r_half].t() 
        fused_down = torch.cat([basis_down_t, mean_down_reshaped], dim=0)
        
        self.register_buffer("fused_down", fused_down.to(storage_dtype).contiguous())
        self.register_buffer("mean_up", mean_up.to(storage_dtype).contiguous())
        self.register_buffer("basis_up", basis_up[:, :self.r_half].to(storage_dtype).contiguous())     

        # --- Trainable Parameters ---
        if shared_alpha_down is not None:
            self.alpha_down = shared_alpha_down
            self.is_alpha_down_shared = True
        else:
            self.alpha_down = nn.Parameter(torch.empty(self.r_half, self.r_half))
            nn.init.normal_(self.alpha_down, std=0.02)
            self.is_alpha_down_shared = False

        self.alpha_up = nn.Parameter(torch.zeros(self.num_experts, self.r_half, self.r_half))

        if self.num_experts > 1:
            self.gate = nn.Linear(self.in_dim, self.num_experts)
            nn.init.normal_(self.gate.weight, std=0.01)
            nn.init.zeros_(self.gate.bias)
            self.register_buffer("expert_mask", torch.ones(self.num_experts))
        else:
            self.gate = None
            self.register_buffer("expert_mask", torch.ones(1))

        # --- Scale ---
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
        loaded_from_cache = False
        if loaded_null_down is not None and loaded_null_up is not None:
            expected_in = self.in_dim if not self.is_conv else ((self.in_dim // self.groups) * self.k_size[0] * self.k_size[1])
            expected_out = self.out_dim
            
            # Check for potential swap and dimension mismatch
            if loaded_null_down.shape[1] != expected_in or loaded_null_up.shape[0] != expected_out:
                if loaded_null_down.shape[0] == expected_out and loaded_null_up.shape[1] == expected_in:
                    loaded_null_down, loaded_null_up = loaded_null_up, loaded_null_down
            
            # Final validation
            if loaded_null_down.shape[1] == expected_in and loaded_null_up.shape[0] == expected_out:
                self.register_buffer("null_down", loaded_null_down[:self.r_half, :].to(storage_dtype).contiguous()) 
                self.register_buffer("null_up", loaded_null_up[:, :self.r_half].to(storage_dtype).contiguous())     
                loaded_from_cache = True
            else:
                logger.warning(f"SVD cache dimension mismatch for {lora_name}. Expected in={expected_in}, out={expected_out}. Got down={loaded_null_down.shape}, up={loaded_null_up.shape}. Recomputing SVD.")
                loaded_null_down = None
                loaded_null_up = None

        if not loaded_from_cache:
            weight_for_svd = get_dequantized_weight(org_module)
            if weight_for_svd is None:
                 raise ValueError(f"Failed to extract weights from {lora_name} for SVD.")
            
            with torch.no_grad():
                u_null, v_null = compute_null_space_basis(
                    weight_for_svd, 
                    self.r_half, 
                    device=weight_for_svd.device
                )
            
            self.register_buffer("null_down", v_null.to(storage_dtype).contiguous()) 
            self.register_buffer("null_up", u_null.to(storage_dtype).contiguous())   
            del weight_for_svd
            torch.cuda.empty_cache()

        # LaLoRA Buffers
        self.register_buffer("lalora_mean_down", lalora_mean_init.to(storage_dtype) if lalora_mean_init is not None else torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))
        self.register_buffer("lalora_mean_up", lalora_mean_init.to(storage_dtype) if lalora_mean_init is not None else torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))
        self.register_buffer("lalora_precision_down", lalora_precision_init.to(storage_dtype) if lalora_precision_init is not None else torch.ones(self.r_half, self.r_half, dtype=storage_dtype))
        self.register_buffer("lalora_precision_up", lalora_precision_init.to(storage_dtype) if lalora_precision_init is not None else torch.ones(self.r_half, self.r_half, dtype=storage_dtype))

        self.use_gradient_checkpointing = False

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
        
        diff_u = self.alpha_up - self.lalora_mean_up.to(device=self.alpha_down.device, dtype=dtype).unsqueeze(0)
        loss_u = torch.sum((diff_u * diff_u) * self.lalora_precision_up.to(device=self.alpha_down.device, dtype=dtype).unsqueeze(0))
        
        return 0.5 * self.lalora_lambda * (loss_d + loss_u)

    def forward(self, x, *args, **kwargs):
        if self.use_gradient_checkpointing and self.training:
            if not x.requires_grad:
                x.requires_grad_(True)
            out = torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            out = self._forward(x)
        return out

    def _forward(self, x):
        dtype = x.dtype
        device = x.device
        
        # --- Cross-Path Architecture (Sandwich) ---
        # Path 1: Frozen Down -> Train Up (EffiLoRA)
        # Path 2: Train Down -> Frozen Up

        # Prepare Inputs
        frozen_down = self.null_down.to(device=device, dtype=dtype)
        frozen_up = self.null_up.to(device=device, dtype=dtype)

        w_fused = self.fused_down.to(device=device, dtype=dtype)
        a_down = self.alpha_down[:self.r_half, :self.r_half].to(device=device, dtype=dtype)
        
        # Prepare Gate Scores for MoE (Used in Path 1 Up-Projection)
        gate_scores = None
        if self.num_experts > 1:
            if self.is_conv:
                gate_input = x.mean(dim=(2, 3))
            else:
                gate_input = x.mean(dim=1) if x.dim() == 3 else x
            
            gate_logits = self.gate(gate_input.to(dtype))
            if self.training:
                mask = self.expert_mask.to(device=device, dtype=dtype).view(1, -1)
                gate_scores = F.softmax(gate_logits, dim=-1) * mask
                gate_scores = gate_scores / (gate_scores.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Prepare Scaling
        s1, s2 = 1.0, 1.0
        if not self.use_dora:
            s = self.s.to(device=device, dtype=dtype)
            s1 = s[:self.r_half].mean() # Simplify to scalar for mixing
            s2 = s[self.r_half:].mean()

        # === Path 1: Frozen Down -> Train Up (MoE) ===
        # 1. Frozen Down
        if self.is_conv:
            fd = frozen_down.view(self.r_half, self.in_dim // self.groups, *self.k_size)
            h1 = F.conv2d(x, fd, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            h1 = F.linear(x, frozen_down)
        
        # Scale intermediate
        if not self.use_dora: h1 = h1 * s1
        
        # 2. Train Up (MoE Mixing + Basis Up Projection)
        # Apply MoE Mixing (Alpha Up)
        a_up_experts = self.alpha_up.to(device=device, dtype=dtype)
        
        if self.num_experts > 1:
            # h1 is (B, R, ...)
            if self.is_conv:
                # Optimized MoE for Conv
                # Weights Mixing
                b, r, h, w = h1.shape
                # gate_scores: (B, E), Experts: (E, R, R) -> (B, R, R)
                mixed_weight = torch.einsum('be, eoi -> boi', gate_scores, a_up_experts)
                h1_flat = h1.view(b, r, -1).permute(0, 2, 1) # (B, HW, R)
                h1_mixed_flat = torch.bmm(h1_flat, mixed_weight.transpose(1, 2)) # (B, HW, R)
                h1_mixed = h1_mixed_flat.permute(0, 2, 1).view(b, r, h, w)
            else:
                # Optimized MoE for Linear
                mixed_weight = torch.einsum('be, eoi -> boi', gate_scores, a_up_experts)
                if h1.dim() == 2:
                    h1_mixed = torch.bmm(h1.unsqueeze(1), mixed_weight.transpose(1, 2)).squeeze(1)
                else:
                    h1_mixed = torch.bmm(h1, mixed_weight.transpose(1, 2))
        else:
            # Single Expert
            a_up = a_up_experts[0]
            if self.is_conv:
                h1_mixed = F.conv2d(h1, a_up.view(self.r_half, self.r_half, 1, 1))
            else:
                h1_mixed = F.linear(h1, a_up)
                
        # 3. Final Basis Projection (Basis Up)
        b_up = self.basis_up.to(device=device, dtype=dtype)
        m_up = self.mean_up.to(device=device, dtype=dtype)
        
        if self.is_conv:
            path1 = F.conv2d(h1_mixed, b_up.view(self.out_dim, self.r_half, 1, 1))
            # Correction from mean (using intermediate sum)
            # This is approximate for efficiency, or we can use the exact formulation
            # w_up_gen = basis @ alpha + mean. 
            # path1 = h1 @ (basis @ alpha + mean).T = h1 @ alpha.T @ basis.T + h1 @ mean.T
            # Current: h1_mixed @ basis.T. We need to add h1 @ mean.T
            h1_sum = h1.sum(dim=1, keepdim=True)
            path1_bias = F.conv2d(h1_sum, m_up.view(self.out_dim, 1, 1, 1))
            path1 = path1 + path1_bias
        else:
            path1 = F.linear(h1_mixed, b_up)
            h1_sum = h1.sum(dim=-1, keepdim=True)
            path1_bias = F.linear(h1_sum, m_up.view(-1, 1))
            path1 = path1 + path1_bias


        # === Path 2: Train Down -> Frozen Up ===
        # 1. Train Down (Fused Basis + Alpha Down)
        if self.is_conv:
            h_fused = F.conv2d(x, w_fused.view(self.r_half + 1, self.in_dim // self.groups, *self.k_size), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            h_in = h_fused.narrow(1, 0, self.r_half)
            bias_down = h_fused.narrow(1, self.r_half, 1)
            h2 = F.conv2d(h_in, a_down.view(self.r_half, self.r_half, 1, 1))
            h2 = h2.add_(bias_down)
        else:
            h_fused = F.linear(x, w_fused)
            if x.dim() == 2:
                h_in = h_fused.narrow(1, 0, self.r_half)
                bias_down = h_fused.narrow(1, self.r_half, 1)
            else:
                h_in = h_fused.narrow(2, 0, self.r_half)
                bias_down = h_fused.narrow(2, self.r_half, 1)
            h2 = F.linear(h_in, a_down)
            h2 = h2.add_(bias_down)
        
        # Scale intermediate
        if not self.use_dora: h2 = h2 * s2

        # 2. Frozen Up
        if self.is_conv:
            fu = frozen_up.view(self.out_dim, self.r_half, 1, 1)
            path2 = F.conv2d(h2, fu)
        else:
            path2 = F.linear(h2, frozen_up)

        # === Combine ===
        lx = path1 + path2
        
        # DoRA Logic
        if self.use_dora:
            m = self.m.to(device=device, dtype=dtype)
            if self.is_conv:
                lx = lx * m.view(1, -1, 1, 1)
            else:
                lx = lx * m 
                
        # --- Apply to Original Output with ABM Support ---
        org_out = self.org_forward(x)
        final_out = org_out + lx * self.multiplier
        
        if self.abm_enabled and self.training:
             current_bs = final_out.shape[0]
             limit = self.abm_batch_size if self.abm_batch_size is not None else current_bs
             limit = min(limit, current_bs)
             abm_out_slice = final_out[:limit]
             abm_org_slice = org_out[:limit]
             
             with torch.no_grad():
                 tau_slice = torch.sign(abm_org_slice)
                 hinge = (-tau_slice * abm_out_slice).add_(self.abm_margin).relu_()
                 loss_val = torch.mean(hinge.pow(2)) * self.abm_weight
                 self.abm_loss = self.abm_loss + loss_val
                 tau_full = torch.zeros_like(org_out, dtype=torch.int8)
                 tau_full[:limit] = tau_slice.to(torch.int8)
             
             final_out = ABMGradInjector.apply(final_out, tau_full, self.abm_margin, self.abm_weight)

        return final_out

# --- Custom Autograd Function for ABM Gradient Injection ---
class ABMGradInjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, final_out, tau_int8, margin, weight):
        tau = tau_int8.to(final_out.dtype)
        hinge = (-tau * final_out).add_(margin).relu_()
        ctx.save_for_backward(hinge, tau_int8)
        ctx.weight = weight
        ctx.numel = final_out.numel()
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        hinge, tau_int8 = ctx.saved_tensors
        tau = tau_int8.to(hinge.dtype)
        scale = 2.0 * ctx.weight / ctx.numel
        abm_grad = hinge.mul_(scale).mul_(-tau)
        if grad_output is not None:
            grad_input = grad_output + abm_grad
        else:
            grad_input = abm_grad
        return grad_input, None, None, None

class NullUniversalNetwork(nn.Module):
    def __init__(self, text_encoder, unet, multiplier=1.0, lalora_lambda=0.0, use_dora=False, enable_effilora=False, num_experts=4, loraplus_lr_ratio=16.0, abm_batch_size=None):
        super().__init__()
        self.multiplier = multiplier
        self.lalora_lambda = lalora_lambda
        self.use_dora = use_dora
        self.enable_effilora = enable_effilora
        self.num_experts = num_experts if enable_effilora else 1
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.abm_batch_size = abm_batch_size
        self.modules_dict = nn.ModuleDict()
        
        self.text_encoder = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.unet = unet
        self.shared_alpha_down = None

    def load_basis_and_init(self, basis_path, lalora_context_path=None, weights_path=None, svd_cache_path=None, total_rank=32, ignore_te=False, storage_dtype=torch.float32):
        logger.info(f"Loading Universal Basis from: {basis_path}")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis file not found: {basis_path}")

        basis_state_dict = load_file(basis_path)
        
        if self.enable_effilora:
            r_half = total_rank // 2
            self.shared_alpha_down = nn.Parameter(torch.empty(r_half, r_half))
            nn.init.normal_(self.shared_alpha_down, std=0.02)
            logger.info(f"[EffiLoRA] Enabled: Using shared alpha_down matrix (Rank={r_half}) with {self.num_experts} Experts.")

        loaded_weights = {}
        if weights_path and os.path.exists(weights_path):
            if weights_path.endswith(".safetensors"):
                loaded_weights = load_file(weights_path)
            else:
                loaded_weights = torch.load(weights_path, map_location="cpu")

        loaded_svd_cache = {}
        if svd_cache_path and os.path.exists(svd_cache_path):
            loaded_svd_cache = load_file(svd_cache_path)

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
                    if self.use_dora: l_scale = find_key(loaded_weights, "m")
                    else: l_scale = find_key(loaded_weights, "s")
                
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
                    num_experts=self.num_experts,
                    abm_batch_size=self.abm_batch_size
                )
                
                if loaded_weights:
                    ad = find_key(loaded_weights, "alpha_down")
                    au = find_key(loaded_weights, "alpha_up")
                    gate_w = find_key(loaded_weights, "gate.weight")
                    gate_b = find_key(loaded_weights, "gate.bias")

                    if self.enable_effilora:
                        shared_ad = loaded_weights.get("shared_alpha_down", None)
                        if shared_ad is not None: self.shared_alpha_down.data.copy_(shared_ad)
                        elif ad is not None and count == 0: self.shared_alpha_down.data.copy_(ad)
                    else:
                        if ad is not None: u_module.alpha_down.data.copy_(ad)

                    if au is not None:
                        if au.shape == u_module.alpha_up.shape: u_module.alpha_up.data.copy_(au)
                        elif au.dim() == 2 and u_module.alpha_up.dim() == 3: u_module.alpha_up.data[0].copy_(au)
                    
                    if u_module.gate is not None and gate_w is not None:
                        u_module.gate.weight.data.copy_(gate_w)
                        if gate_b is not None: u_module.gate.bias.data.copy_(gate_b)
                
                self.modules_dict[safe_name] = u_module
                count += 1
        
        del basis_state_dict
        if loaded_weights: del loaded_weights
        if loaded_svd_cache: del loaded_svd_cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()
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
            
    def update_effilora_mask(self):
        if not self.enable_effilora or self.num_experts <= 1: return
        with torch.no_grad():
            for module in self.modules_dict.values():
                norms = module.alpha_up.norm(p=2, dim=(1, 2))
                k = max(1, int(self.num_experts * 0.5)) 
                _, topk_indices = torch.topk(norms, k)
                mask = torch.zeros_like(norms)
                mask[topk_indices] = 1.0
                module.expert_mask.copy_(mask)

    def on_step_start(self, text_encoder, unet):
        self.update_effilora_mask()

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr, loraplus_lr_ratio=None):
        self.requires_grad_(False)
        te_params_low, te_params_high, unet_params_low, unet_params_high = [], [], [], []
        added_shared_down = False
        safe_default = default_lr if default_lr is not None else 1e-4
        te_lr_base = text_encoder_lr if text_encoder_lr is not None else safe_default
        unet_lr_base = unet_lr if unet_lr is not None else safe_default
        if loraplus_lr_ratio is None: loraplus_lr_ratio = self.loraplus_lr_ratio if self.loraplus_lr_ratio else 1.0

        for name, module in self.modules_dict.items():
            if module.is_alpha_down_shared:
                if not added_shared_down:
                    module.alpha_down.requires_grad_(True)
                    unet_params_low.append(module.alpha_down) 
                    added_shared_down = True
            else:
                module.alpha_down.requires_grad_(True)
                if "text_encoder" in name or "te1" in name or "te2" in name: te_params_low.append(module.alpha_down)
                else: unet_params_low.append(module.alpha_down)

            module.alpha_up.requires_grad_(True)
            if module.gate is not None: module.gate.requires_grad_(True)
            if self.use_dora: module.m.requires_grad_(True)
            else: module.s.requires_grad_(True)
            
            is_te = "te1" in name or "te2" in name or "text_encoder" in name
            target_low = te_params_low if is_te else unet_params_low
            target_high = te_params_high if is_te else unet_params_high
            
            if self.use_dora: target_low.append(module.m)
            else: target_low.append(module.s)
            target_high.append(module.alpha_up)
            if module.gate is not None: target_high.append(module.gate.weight)
            if module.gate is not None and module.gate.bias is not None: target_high.append(module.gate.bias)

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
            if self.use_dora: module.m.requires_grad_(True)
            else: module.s.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def enable_gradient_checkpointing(self):
        for module in self.modules_dict.values():
            module.use_gradient_checkpointing = True
        
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for module in self.modules_dict.values():
            module.multiplier = multiplier

    def get_regularization_loss(self):
        total_reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.lalora_lambda <= 0.0: return total_reg_loss
        for module in self.modules_dict.values():
            total_reg_loss += module.get_regularization_loss()
        return total_reg_loss

    def enable_abm(self, enabled=True, margin=0.5):
        logger.info(f"Setting ABM mode to {enabled} (margin={margin})")
        for module in self.modules_dict.values():
            module.abm_enabled = enabled
            module.abm_margin = margin
            module.abm_loss = 0.0 

    def get_abm_loss(self):
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        for module in self.modules_dict.values():
            if torch.is_tensor(module.abm_loss):
                total_loss = total_loss + module.abm_loss.to(device=device, dtype=torch.float32)
            module.abm_loss = 0.0
        return total_loss

    def set_abm_weights(self, strategy='uniform'):
        modules = list(self.modules_dict.values())
        L = len(modules)
        for i, module in enumerate(modules):
            if strategy == 'uniform': w = 1.0
            elif strategy == 'linear': w = (i + 1) / L
            elif strategy == 'quadratic': w = ((i + 1) / L) ** 2
            else: w = 1.0
            module.abm_weight = w

    def save_weights(self, file, dtype, metadata):
        if metadata is None: metadata = {}
        metadata.update({
            "lalora_lambda": str(self.lalora_lambda),
            "network_dim": str(next(iter(self.modules_dict.values())).r_half * 2),
            "lora_plus": "true",
            "use_dora": str(self.use_dora).lower(),
            "effilora": str(self.enable_effilora).lower(),
            "num_experts": str(self.num_experts),
            "loraplus_lr_ratio": str(self.loraplus_lr_ratio),
            "abm_supported": "true"
        })
        
        state_dict = {}
        def to_cpu(t): return t.detach().cpu().to(dtype if dtype else torch.float16).contiguous()

        if self.enable_effilora and self.shared_alpha_down is not None:
            state_dict["shared_alpha_down"] = to_cpu(self.shared_alpha_down)

        for name, module in self.modules_dict.items():
            state_dict[f"{name}.alpha_down"] = to_cpu(module.alpha_down)
            state_dict[f"{name}.alpha_up"] = to_cpu(module.alpha_up)
            if module.gate is not None:
                state_dict[f"{name}.gate.weight"] = to_cpu(module.gate.weight)
                if module.gate.bias is not None: state_dict[f"{name}.gate.bias"] = to_cpu(module.gate.bias)
            if self.use_dora: state_dict[f"{name}.m"] = to_cpu(module.m)
            else: state_dict[f"{name}.s"] = to_cpu(module.s)
            state_dict[f"{name}.null_down"] = to_cpu(module.null_down)
            state_dict[f"{name}.null_up"] = to_cpu(module.null_up)
        
        if os.path.splitext(file)[1] == ".safetensors": save_file(state_dict, file, metadata)
        else: torch.save(state_dict, file)

    def save_svd_cache(self, file):
        logger.info(f"Saving SVD cache to {file}...")
        state_dict = {}
        dtype = torch.float32 
        for name, module in self.modules_dict.items():
            state_dict[f"{name}.null_down"] = module.null_down.detach().clone().to(dtype).contiguous()
            state_dict[f"{name}.null_up"] = module.null_up.detach().clone().to(dtype).contiguous()
        if os.path.splitext(file)[1] == ".safetensors": save_file(state_dict, file)
        else: torch.save(state_dict, file)

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    lalora_lambda = float(kwargs.get("lalora_lambda", 0.0))
    basis_path = kwargs.get("basis_path", None)
    lalora_context_path = kwargs.get("lalora_context_path", None)
    svd_cache_path = kwargs.get("svd_cache_path", None) 
    use_dora = str(kwargs.get("use_dora", False)).lower() == "true"
    enable_effilora = str(kwargs.get("enable_effilora", False)).lower() == "true"
    num_experts = int(kwargs.get("num_experts", 4))
    loraplus_lr_ratio = float(kwargs.get("loraplus_lr_ratio", 16.0))
    abm_batch_size = kwargs.get("abm_batch_size", None)
    if abm_batch_size is not None: abm_batch_size = int(abm_batch_size)
    network_weights = kwargs.get("network_weights", kwargs.get("weights", None))
    ignore_te = str(kwargs.get("ignore_te", False)).lower() == "true"

    if basis_path is None: raise ValueError("Null-Universal Network requires 'basis_path'")
    if network_dim is None: network_dim = 32 
    
    network = NullUniversalNetwork(
        text_encoder, unet, 
        multiplier=multiplier, 
        lalora_lambda=lalora_lambda,
        use_dora=use_dora,
        enable_effilora=enable_effilora,
        num_experts=num_experts,
        loraplus_lr_ratio=loraplus_lr_ratio,
        abm_batch_size=abm_batch_size
    )
    
    storage_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported(): storage_dtype = torch.bfloat16
        else: storage_dtype = torch.float16
    
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
    ignore_te = str(kwargs.get("ignore_te", False)).lower() == "true"
    if basis_path is None: raise ValueError("Null-Universal Network requires 'basis_path' even when loading weights")
    use_dora = str(kwargs.get("use_dora", False)).lower() == "true"
    enable_effilora = str(kwargs.get("enable_effilora", False)).lower() == "true"
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
        if torch.cuda.is_bf16_supported(): storage_dtype = torch.bfloat16
        else: storage_dtype = torch.float16

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