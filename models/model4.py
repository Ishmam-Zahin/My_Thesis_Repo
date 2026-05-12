import torch
from torch import nn

class VideoFeatureExtractor(nn.Module):
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, weight_path=None):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            vit_state_dict = {k.replace("vit.", "", 1): v for k, v in state_dict.items() if k.startswith("vit.")}
            missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)

        for p in self.vit.parameters():
            p.requires_grad = False
        for block in self.vit.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
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
        cls_tokens = features['x_norm_clstoken']
        return cls_tokens.reshape(B, T, self.D)


class MyModel(nn.Module):
    """
    Version 1: Query Attention Pooling (no extra CLS token)
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim

        self.vit = VideoFeatureExtractor(vit_name=vit_name, weight_path=vit_weight_path)

        # Learnable query token for pooling
        self.query_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Positional embedding for the 8 frames only
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_of_frames, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Cross-attention pooling
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cls_features = self.vit(x)                                   # [B, 8, D]
        
        # Add positional embedding to frames
        cls_features = cls_features + self.temporal_pos_embed
        
        # Learnable query
        query = self.query_token.expand(B, 1, -1)                    # [B, 1, D]
        
        # Cross-attention: query attends to the 8 frame tokens
        attn_out, _ = self.cross_attn(query, cls_features, cls_features)
        
        video_feature = attn_out.squeeze(1)                          # [B, D]
        
        video_feature = self.norm(video_feature)
        logits = self.classifier(video_feature)
        return logits