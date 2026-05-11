import torch
from torch import nn


class VideoFeatureExtractor(nn.Module):
    """
    DINOv2 ViT that returns ONLY the CLS tokens (patches are completely ignored).
    - Last 2 transformer blocks + ALL LayerNorms are unfrozen
    """
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, weight_path=None):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)

        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            vit_state_dict = {
                k.replace("vit.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("vit.")
            }
            missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False

        # Unfreeze only the last 2 transformer blocks
        for block in self.vit.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        # Unfreeze ALL LayerNorm layers
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True

        self.vit.eval()
        self.D = feature_dim

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.vit.forward_features(x_flat)
        cls_tokens = features['x_norm_clstoken']          # [B*T, 384]
        return cls_tokens.reshape(B, T, self.D)           # [B, T, D]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=False
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class MyModel(nn.Module):
    """
    Clean final version:
    - Only uses 8 CLS tokens from DINOv2
    - Adds one learnable video-level CLS token
    - Uses temporal Transformer blocks
    - Returns ONLY logits (no mincut_loss, no ortho_loss)
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim

        self.vit = VideoFeatureExtractor(
            vit_name=vit_name,
            weight_path=vit_weight_path
        )

        # Learnable video-level CLS token
        self.class_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Temporal positional embedding for 9 tokens (video_CLS + 8 frame_CLS)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_of_frames, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Temporal Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=feature_dim,
                heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])

        # Final layers
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # Get 8 frame-level CLS tokens
        cls_features = self.vit(x)                          # [B, 8, D]

        # Prepend video-level CLS token
        video_cls = self.class_token.expand(B, 1, -1)       # [B, 1, D]
        sequence = torch.cat([video_cls, cls_features], dim=1)  # [B, 9, D]

        # Add temporal positional embedding
        sequence = sequence + self.temporal_pos_embed

        # Temporal modeling
        for block in self.transformer_blocks:
            sequence = block(sequence)

        # Take the final video-level CLS token
        video_feature = sequence[:, 0]                      # [B, D]

        # Classification
        video_feature = self.norm(video_feature)
        logits = self.classifier(video_feature)

        return logits                                        # ← Only logits now