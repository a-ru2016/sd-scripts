import torch
import torch.nn as nn
from networks import lora

class VDropNetwork(lora.LoRANetwork):
    def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4, alpha=1, dropout=None, rank_dropout=None, module_class=lora.LoRAModule, varbose=False, **kwargs):
        super().__init__(text_encoder, unet, multiplier, lora_dim, alpha, dropout, rank_dropout, module_class=module_class, varbose=varbose)
        
        # DINOv2 (1024) -> SDXL Text Embedding (2048: 768 + 1280)
        # 簡易的なMLP Projector
        input_dim = kwargs.get("dino_dim", 1024)
        output_dim = kwargs.get("text_dim", 2048) # SDXL default combined
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Projectorの初期化 (IP-Adapterなどを参考に)
        self.init_projector_weights()

    def init_projector_weights(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_projector(self, visual_embeds):
        """
        visual_embeds: (Batch, DINO_Dim)
        Returns: (Batch, 1, Text_Dim) -> Token sequence length 1
        """
        # dtype conversion
        dtype = self.projector[0].weight.dtype
        device = self.projector[0].weight.device
        visual_embeds = visual_embeds.to(device=device, dtype=dtype)

        visual_tokens = self.projector(visual_embeds)
        return visual_tokens.unsqueeze(1) # (B, 1, Dim)

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1
    if network_alpha is None:
        network_alpha = network_dim

    network = VDropNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        varbose=True,
        **kwargs
    )
    return network
