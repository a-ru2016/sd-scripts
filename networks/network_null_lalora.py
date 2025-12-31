import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
import logging

# --- QLoRA Support: Import bitsandbytes if available ---
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

logger = logging.getLogger(__name__)

# --- Helper Functions for Null-LoRA ---

def get_dequantized_weight(module):
    """
    QLoRA対応: モジュールが量子化されている場合、一時的に非量子化してfloat32の重みを返す。
    通常の層であればそのまま重みを返す。
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
    事前学習済み重みのSVDを行い、Null Space基底 (U_null, V_null) を抽出する。
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

    u_null = U[:, -rank:].clone()
    v_null = Vh[-rank:, :].clone()

    return u_null, v_null

class NullUniversalModule(nn.Module):
    def __init__(
        self, 
        lora_name, 
        org_module, 
        # Universal Basis (Trainable側生成用)
        mean_down, basis_down, 
        mean_up, basis_up,
        # Config
        multiplier=1.0, 
        total_rank=32,
        lalora_lambda=0.0,
        # LaLoRA Init
        lalora_mean_init=None,
        lalora_precision_init=None,
        # Pre-calculated Null Space (Load時用)
        loaded_null_down=None,
        loaded_null_up=None,
        loaded_scale=None,
        # Memory Optimization
        storage_dtype=torch.float32
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.org_module = org_module
        self.lalora_lambda = lalora_lambda
        
        # Null-LoRAはランクを半分に分割する (Half Trainable, Half Frozen)
        self.r_half = total_rank // 2
        if self.r_half < 1: self.r_half = 1
        
        # --- 1. Universal Basis (Trainable側) ---
        # basisのshapeは (dim, basis_rank)
        # ここで basis_rank を self.r_half に制限して使用する
        self.register_buffer("mean_down", mean_down.to(storage_dtype))
        self.register_buffer("basis_down", basis_down[:, :self.r_half].to(storage_dtype)) 
        self.register_buffer("mean_up", mean_up.to(storage_dtype))
        self.register_buffer("basis_up", basis_up[:, :self.r_half].to(storage_dtype))     

        # --- 2. Trainable Parameters (Alpha & Scale) ---
        # Alphaは学習パラメータなのでfloat32推奨だが、Mixed Precision学習時はTrainerがキャストしてくれる
        self.alpha_down = nn.Parameter(torch.zeros(self.r_half, self.r_half))
        self.alpha_up = nn.Parameter(torch.zeros(self.r_half, self.r_half))
        
        # Scale parameter
        if loaded_scale is not None:
             self.s = nn.Parameter(loaded_scale)
        else:
             self.s = nn.Parameter(torch.ones(total_rank))

        # --- Architecture Handling ---
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

        # --- 3. Frozen Null Space Basis ---
        # SVDキャッシュまたは再開用重みから読み込み
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
            # キャッシュがない場合は計算
            weight_for_svd = get_dequantized_weight(org_module)
            if weight_for_svd is None:
                 raise ValueError(f"Failed to extract weights from {lora_name} for SVD.")
            with torch.no_grad():
                u_null, v_null = compute_null_space_basis(
                    weight_for_svd, 
                    self.r_half, 
                    device=weight_for_svd.device
                )
            del weight_for_svd
            torch.cuda.empty_cache()
            
            # v_null is (R, In), u_null is (Out, R)
            self.register_buffer("null_down", v_null.to(storage_dtype)) 
            self.register_buffer("null_up", u_null.to(storage_dtype))   

        # --- 4. LaLoRA Regularization Buffers ---
        if lalora_mean_init is not None:
            if lalora_mean_init.dim() == 1:
                logger.warning(f"LaLoRA context mismatch for {lora_name}: resetting mean/precision to zeros/ones.")
                self.register_buffer("lalora_mean_down", torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))
                self.register_buffer("lalora_mean_up", torch.zeros(self.r_half, self.r_half, dtype=storage_dtype))
            else:
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
        
        basis_d_t = self.basis_down.to(device=device, dtype=dtype).t()
        w_down_generated = (alpha_d @ basis_d_t) + self.mean_down.to(device=device, dtype=dtype).view(1, -1)
        w_up_generated = (self.basis_up.to(device=device, dtype=dtype) @ alpha_u) + self.mean_up.to(device=device, dtype=dtype).view(-1, 1)

        B_train = w_down_generated.view(self.down_shape_train) 
        A_train = w_up_generated.view(self.up_shape_train)     
        
        if self.is_conv:
            frozen_down = self.null_down.to(device=device, dtype=dtype).view(self.down_shape_frozen)
            frozen_up = self.null_up.to(device=device, dtype=dtype).view(self.up_shape_frozen)
        else:
            frozen_down = self.null_down.to(device=device, dtype=dtype)
            frozen_up = self.null_up.to(device=device, dtype=dtype)

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
        return self.org_forward(x) + lx * self.multiplier


class NullUniversalNetwork(nn.Module):
    def __init__(self, text_encoder, unet, multiplier=1.0, lalora_lambda=0.0):
        super().__init__()
        self.multiplier = multiplier
        self.lalora_lambda = lalora_lambda
        self.modules_dict = nn.ModuleDict()
        
        self.text_encoder = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.unet = unet

    def load_basis_and_init(self, basis_path, lalora_context_path=None, weights_path=None, svd_cache_path=None, total_rank=32, ignore_te=False, storage_dtype=torch.float32):
        logger.info(f"Loading Universal Basis from: {basis_path}")
        if not os.path.exists(basis_path):
            raise FileNotFoundError(f"Basis file not found: {basis_path}")

        basis_state_dict = load_file(basis_path)
        
        lalora_context = {}
        if lalora_context_path and os.path.exists(lalora_context_path):
            logger.info(f"Loading LaLoRA context from: {lalora_context_path}")
            lalora_context = load_file(lalora_context_path)
            
        loaded_weights = {}
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading existing Null-LoRA weights from: {weights_path}")
            if weights_path.endswith(".safetensors"):
                loaded_weights = load_file(weights_path)
            else:
                loaded_weights = torch.load(weights_path, map_location="cpu")

        # SVD Cache Loading
        loaded_svd_cache = {}
        if svd_cache_path and os.path.exists(svd_cache_path):
            logger.info(f"Loading SVD cache from: {svd_cache_path}")
            loaded_svd_cache = load_file(svd_cache_path)

        # Detect Basis Keys
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
        logger.info(f"Initializing Null-Universal Modules with Rank={total_rank} (Train: {total_rank//2}, Frozen: {total_rank//2})...")
        logger.info(f"Buffers will be stored in: {storage_dtype}")
        
        for base_name, parts in layer_groups.items():
            if "down" not in parts or "up" not in parts:
                continue

            is_te = "te1" in base_name or "te2" in base_name or "text_encoder" in base_name
            if ignore_te and is_te:
                continue

            target_module = self.find_target_module(base_name)
            
            if target_module is not None:
                mean_d = basis_state_dict[parts["down"] + ".mean"]
                basis_d = basis_state_dict[parts["down"] + ".basis"]
                mean_u = basis_state_dict[parts["up"] + ".mean"]
                basis_u = basis_state_dict[parts["up"] + ".basis"]

                safe_name = base_name.replace(".", "_")
                
                l_mean = None
                l_prec = None
                l_null_d = None
                l_null_u = None
                l_scale = None
                
                # Priority 1: Loaded Weights (for Resume/Fine-tuning)
                if loaded_weights:
                    # Check for non-prefixed keys (saved by this script)
                    nd_key = f"{safe_name}.null_down"
                    nu_key = f"{safe_name}.null_up"
                    scale_key = f"{safe_name}.s"
                    
                    # Check for prefixed keys (standard state_dict)
                    p_nd_key = f"modules_dict.{safe_name}.null_down"
                    p_nu_key = f"modules_dict.{safe_name}.null_up"
                    p_scale_key = f"modules_dict.{safe_name}.s"

                    if nd_key in loaded_weights: l_null_d = loaded_weights[nd_key]
                    elif p_nd_key in loaded_weights: l_null_d = loaded_weights[p_nd_key]

                    if nu_key in loaded_weights: l_null_u = loaded_weights[nu_key]
                    elif p_nu_key in loaded_weights: l_null_u = loaded_weights[p_nu_key]

                    if scale_key in loaded_weights: l_scale = loaded_weights[scale_key]
                    elif p_scale_key in loaded_weights: l_scale = loaded_weights[p_scale_key]
                
                # Priority 2: SVD Cache
                if l_null_d is None and loaded_svd_cache:
                    nd_key = f"{safe_name}.null_down"
                    nu_key = f"{safe_name}.null_up"
                    if nd_key in loaded_svd_cache: l_null_d = loaded_svd_cache[nd_key]
                    if nu_key in loaded_svd_cache: l_null_u = loaded_svd_cache[nu_key]

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
                    storage_dtype=storage_dtype
                )
                
                # Alpha loading
                if loaded_weights:
                    ad_key = f"{safe_name}.alpha_down"
                    au_key = f"{safe_name}.alpha_up"
                    p_ad_key = f"modules_dict.{safe_name}.alpha_down"
                    p_au_key = f"modules_dict.{safe_name}.alpha_up"

                    if ad_key in loaded_weights: u_module.alpha_down.data.copy_(loaded_weights[ad_key])
                    elif p_ad_key in loaded_weights: u_module.alpha_down.data.copy_(loaded_weights[p_ad_key])

                    if au_key in loaded_weights: u_module.alpha_up.data.copy_(loaded_weights[au_key])
                    elif p_au_key in loaded_weights: u_module.alpha_up.data.copy_(loaded_weights[p_au_key])
                
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
            
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(False)
        params = []
        for module in self.modules_dict.values():
            module.alpha_down.requires_grad_(True)
            module.alpha_up.requires_grad_(True)
            module.s.requires_grad_(True)
            params.append(module.alpha_down)
            params.append(module.alpha_up)
            params.append(module.s)
        return [{"params": params, "lr": default_lr}], ["alpha_down", "alpha_up", "s"]

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(False)
        for module in self.modules_dict.values():
            module.alpha_down.requires_grad_(True)
            module.alpha_up.requires_grad_(True)
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

    # --- Added Method: load_weights for sd-scripts compatibility ---
    def load_weights(self, file):
        """
        sd-scripts (train_network.py) から呼び出されるメソッド。
        --network_weights 引数で指定されたファイルを読み込む。
        """
        logger.info(f"Loading weights from {file}")
        if file.endswith(".safetensors"):
            weights = load_file(file)
        else:
            weights = torch.load(file, map_location="cpu")

        # 読み込み用State Dictの準備
        # このスクリプトの save_weights は "modules_dict." プレフィックスを削除して保存するが、
        # PyTorchの load_state_dict は内部構造 (modules_dict.xxx) と一致するキーを期待する。
        # また、Acceleratorで保存された完全なチェックポイントは "modules_dict." を持っている可能性がある。
        # 両方に対応する。
        
        my_state_dict = {}
        for key, value in weights.items():
            if key.startswith("modules_dict."):
                my_state_dict[key] = value
            else:
                # プレフィックスがない場合、付与してマッピング
                my_state_dict[f"modules_dict.{key}"] = value
        
        # 厳密な一致(strict=True)だと、metadata等が含まれている場合にエラーになることがあるため、
        # 必要なキーだけ読み込まれることを期待して strict=False にする手もあるが、
        # 構造が変わっていない限りは基本ロードできるはず。
        # ここでは安全のため missing_keys のみ許容し、size mismatch はエラーにする標準挙動を利用。
        info = self.load_state_dict(my_state_dict, strict=False)
        logger.info(f"Weights loaded: {info}")
        
        # Buffer類 (null_down/up) も state_dict に含まれていれば更新される。

    def save_weights(self, file, dtype, metadata):
        if metadata is None: metadata = {}
        metadata["lalora_lambda"] = str(self.lalora_lambda)
        metadata["network_dim"] = str(next(iter(self.modules_dict.values())).r_half * 2)
        
        state_dict = {}
        for name, module in self.modules_dict.items():
            state_dict[f"{name}.alpha_down"] = module.alpha_down.detach().clone().to(dtype if dtype else torch.float16).contiguous()
            state_dict[f"{name}.alpha_up"] = module.alpha_up.detach().clone().to(dtype if dtype else torch.float16).contiguous()
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
    
    # Check for weights passed via arguments (resume from specific file)
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
    
    network = NullUniversalNetwork(text_encoder, unet, multiplier=multiplier, lalora_lambda=lalora_lambda)
    
    storage_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            storage_dtype = torch.bfloat16
        else:
            storage_dtype = torch.float16
    
    # Initialize
    # weights_path をここで渡すことで、SVD計算をスキップして初期値をロードできる
    network.load_basis_and_init(
        basis_path, 
        lalora_context_path, 
        weights_path=network_weights, # Added
        total_rank=network_dim,
        svd_cache_path=svd_cache_path,
        ignore_te=ignore_te,          
        storage_dtype=storage_dtype   
    )
    
    if svd_cache_path is not None and not os.path.exists(svd_cache_path) and network_weights is None:
        # 重みをロードしていない（新規計算した）場合のみキャッシュを保存
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
    
    network = NullUniversalNetwork(text_encoder, unet, multiplier=multiplier, lalora_lambda=lalora_lambda)
    
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