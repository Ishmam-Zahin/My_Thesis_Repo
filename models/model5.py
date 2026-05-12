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
    Version 2: BiLSTM temporal aggregator
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim

        self.vit = VideoFeatureExtractor(vit_name=vit_name, weight_path=vit_weight_path)

        # Positional embedding for 8 frames
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_of_frames, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cls_features = self.vit(x)                                   # [B, 8, D]
        cls_features = cls_features + self.temporal_pos_embed       # positional
        
        lstm_out, _ = self.bilstm(cls_features)                      # [B, 8, feature_dim]
        
        # Mean pooling across time
        video_feature = lstm_out.mean(dim=1)                         # [B, D]
        
        video_feature = self.norm(video_feature)
        logits = self.classifier(video_feature)
        return logits