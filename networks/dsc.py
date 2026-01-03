import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network_utils import LoRAModule

class DSCModule(LoRAModule):
    """
    Dynamic Subspace Composition (DSC) Module
    Paper: https://arxiv.org/abs/2512.23448
    """
    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, 
                 lora_dim=4, alpha=1, dropout=None, rank_dropout=None, module_dropout=None,
                 num_basis=1024, # M: Basis Bank Size (論文推奨: >1000)
                 active_basis=16, # K: Active sparsity (論文推奨: 4~16)
                 router_scale=1.0):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, dropout, rank_dropout, module_dropout)
        
        self.num_basis = num_basis
        self.active_basis = active_basis
        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features
        
        # --- Basis Bank (U, V) ---
        # 論文[cite: 56]: U, V are learnable parameter matrices.
        # basis_u: (M, d_in)  -> Down projection atoms
        # basis_v: (M, d_out) -> Up projection atoms
        self.basis_u = nn.Parameter(torch.randn(num_basis, self.in_dim))
        self.basis_v = nn.Parameter(torch.zeros(num_basis, self.out_dim)) # Initialize V to zero for stability
        
        # --- Router ---
        # 論文[cite: 70]: W_r in R^{d x M}
        self.router = nn.Linear(self.in_dim, num_basis, bias=False)
        
        # --- Hyperparams ---
        self.tau = 10.0 # Clamping threshold
        self.eps = 1e-6
        
        # SVD Initialization Hook (後で呼び出す)
        self.initialized_with_svd = False

    def apply_svd_initialization(self, org_weight):
        """
        SVDによるWarm Start。
        Illustriousの重み(org_weight)を分解し、主要成分をBasisに注入する。
        """
        print(f"Applying SVD Init to {self.lora_name}...")
        with torch.no_grad():
            # org_weight: (out, in) -> SVD -> U(out, out), S(min), Vh(in, in)
            # PyTorch SVD: U, S, Vh
            u, s, vh = torch.linalg.svd(org_weight.float(), full_matrices=False)
            
            # 使用可能なランク数 (Basis数か元のランクの小さい方)
            k = min(self.num_basis, s.shape[0])
            
            # SVDの成分をBasisに移植
            # basis_u (Down) <= Vhの行 (in_dim)
            # basis_v (Up)   <= Uの列 * S (out_dim)
            
            # 注意: DSCの数式的には u_j * v_j なので、特異値 sqrt(S) を両方に分散させるか、片方に寄せる。
            # ここでは一般的なLoRA初期化に倣い、V (Up) 側に重みを寄せるが、
            # DSCは正規化が入るため、方向ベクトルとして重要。
            
            self.basis_u.data[:k] = vh[:k] 
            # basis_vには U * S を入れる
            self.basis_v.data[:k] = (u[:, :k] @ torch.diag(s[:k])).T
            
            # 残りのBasisはノイズのまま (探索用)
            self.initialized_with_svd = True

    def _normalize_basis(self, basis):
        """
        論文[cite: 57]: l2-Projected Normalization
        u_j = hat_u_j / max(epsilon, ||hat_u_j||)
        """
        norms = basis.norm(p=2, dim=1, keepdim=True)
        return basis / torch.max(torch.tensor(self.eps, device=basis.device), norms)

    def forward(self, x):
        # x: (Batch, Seq, In_Dim)
        
        # 1. Project Normalized Basis (On-the-fly normalization)
        # 勾配爆発を防ぐため、Forward毎に射影する [cite: 57]
        u_norm = self._normalize_basis(self.basis_u)
        v_norm = self._normalize_basis(self.basis_v)

        # 2. Routing & Gating
        # [cite: 178] Router Input Normalization (optional but recommended in algo 2)
        x_norm = F.layer_norm(x, (self.in_dim,))
        
        # [cite: 70] Logits calculation & Clamping
        router_logits = self.router(x_norm) # (B, S, M)
        router_logits = torch.clamp(router_logits, -self.tau, self.tau)
        
        # [cite: 71] Softplus
        alpha = F.softplus(router_logits)
        
        # [cite: 183] Top-K Selection
        # topk_vals: (B, S, K), topk_indices: (B, S, K)
        topk_vals, topk_indices = torch.topk(alpha, self.active_basis, dim=-1)
        
        # [cite: 184] Signal Strength S
        S = topk_vals.sum(dim=-1, keepdim=True) # (B, S, 1)
        
        #  Magnitude-Gated Simplex Coefficients
        # z_hat = (alpha / (S + eps)) * tanh(S)
        mixing_coeffs = (topk_vals / (S + self.eps)) * torch.tanh(S) # (B, S, K)

        # 3. Factorized Computation [cite: 125]
        # 行列を復元せず、分解したまま計算する (メモリ効率化)
        
        # Gather active bases
        # shape変更: (B, S, K, D) になるようにgatherする
        # ここはPyTorchのgather実装が少し複雑になるため、einsum的なアプローチをとるか、
        # あるいはバッチ次元をフラットにして計算する。
        
        # 簡易実装: Basisが大きいので、x @ U.T 全体計算は避ける。
        # 選択されたBasisだけを取り出す。
        
        batch_size, seq_len, _ = x.shape
        flat_indices = topk_indices.view(-1) # (B*S*K)
        
        # ActiveなUとVを取得
        active_u = u_norm.index_select(0, flat_indices).view(batch_size, seq_len, self.active_basis, self.in_dim)
        active_v = v_norm.index_select(0, flat_indices).view(batch_size, seq_len, self.active_basis, self.out_dim)
        
        # (x * U_active^T)
        # x: (B, S, 1, In) * active_u: (B, S, K, In) -> sum dim -1 -> (B, S, K)
        c_lat = (x.unsqueeze(2) * active_u).sum(dim=-1)
        
        # Mixing
        c_mix = c_lat * mixing_coeffs # (B, S, K)
        
        # Output Construction
        # c_mix: (B, S, K, 1) * active_v: (B, S, K, Out) -> sum dim 2 -> (B, S, Out)
        y_dyn = (c_mix.unsqueeze(-1) * active_v).sum(dim=2)
        
        # 4. Aux Losses Calculation (Hook用に保存、あるいは戻り値に含める)
        # ここではselfに保存して外部から回収する設計にする
        if self.training:
            self.last_router_logits = router_logits
            self.last_topk_indices = topk_indices
            self.last_u_norm = u_norm
            self.last_v_norm = v_norm
            
        return y_dyn * self.multiplier

    def get_aux_losses(self):
        """
        論文 [cite: 120] に基づく正則化項の計算
        """
        if not hasattr(self, 'last_router_logits'):
            return 0.0
            
        # 1. Load Balancing (Aux Loss) [cite: 103]
        #  Batch全体の分布が均一になるように
        probs = F.softmax(self.last_router_logits, dim=-1)
        mean_probs = probs.mean(dim=(0, 1)) # Batch, Seq平均
        loss_aux = self.num_basis * (mean_probs ** 2).sum()
        
        # 2. Frame Potential (Basis Diversity) 
        #  基底の直交化 (計算コストが高いので、ランダムサンプリングするか、頻度を下げるのが吉)
        #  ここでは厳密解を書くが、実運用では間引くべき
        def frame_potential(basis):
            gram = basis @ basis.T
            # 対角成分以外を最小化
            gram_off_diag = gram - torch.diag(torch.diag(gram))
            return (gram_off_diag ** 2).sum()
            
        loss_frame = frame_potential(self.last_u_norm) + frame_potential(self.last_v_norm)

        # 3. Z-Loss [cite: 118]
        log_z = torch.logsumexp(self.last_router_logits, dim=-1)
        loss_z = (log_z ** 2).mean()

        return {
            "loss_aux": loss_aux,
            "loss_frame": loss_frame,
            "loss_z": loss_z
        }