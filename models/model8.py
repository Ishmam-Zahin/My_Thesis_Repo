import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from einops import rearrange, repeat


# ============================================================
#  1. SEMANTIC KNN (computed per-sample, not static grid KNN)
# ============================================================

def build_semantic_knn_edges(patch_feats: torch.Tensor, K: int = 8) -> torch.Tensor:
    """
    Build KNN graph based on FEATURE SIMILARITY rather than spatial distance.
    This ensures edges connect semantically related patches (e.g., face regions).
    
    Args:
        patch_feats: [N, D]  — patch tokens for one frame/graph
        K: number of neighbors
    Returns:
        edge_index: [2, E]
    """
    N = patch_feats.shape[0]
    # Cosine similarity
    norm = F.normalize(patch_feats, dim=-1)          # [N, D]
    sim = norm @ norm.T                               # [N, N]
    sim.fill_diagonal_(-float('inf'))
    
    _, topk_idx = torch.topk(sim, k=K, dim=-1)       # [N, K]
    
    src = torch.arange(N, device=patch_feats.device).unsqueeze(1).expand(-1, K).reshape(-1)
    dst = topk_idx.reshape(-1)
    
    # Make undirected
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ], dim=0)
    
    return edge_index


def build_temporal_edges(T: int, N: int, device: torch.device) -> torch.Tensor:
    """
    Forward + backward temporal edges between same patch index across frames.
    Bidirectional gives the GNN context in both directions.
    """
    src_list, dst_list = [], []
    patch_idx = torch.arange(N, device=device)
    
    for t in range(T - 1):
        offset_curr = t * N
        offset_next = (t + 1) * N
        # Forward
        src_list.append(patch_idx + offset_curr)
        dst_list.append(patch_idx + offset_next)
        # Backward
        src_list.append(patch_idx + offset_next)
        dst_list.append(patch_idx + offset_curr)
    
    return torch.stack([
        torch.cat(src_list),
        torch.cat(dst_list)
    ], dim=0)


# ============================================================
#  2. PATCH TOKEN PROCESSING MODULES
# ============================================================

class PatchArtifactDetector(nn.Module):
    """
    Directly exploits patch tokens for local artifact detection.
    
    Key insight: DINOv2 patch tokens at different layers capture:
      - Early layers: low-level texture, frequency artifacts (GAN fingerprints)
      - Late layers:  semantic content, face structure
    
    We use spatial attention over patch tokens to find inconsistent regions.
    """
    def __init__(self, feature_dim: int = 384, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Spatial self-attention within each frame
        # Finds patches that are inconsistent with their neighbors
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)
        
        # Per-patch artifact scoring (binary: real/fake patch)
        # This creates a spatial heatmap of potential manipulations
        self.artifact_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Weighted aggregation using artifact scores as attention weights
        self.patch_aggregator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: [B, T, N, D]  where N=256 patches per frame
        Returns:
            frame_feats:    [B, T, D]   — per-frame aggregated features
            artifact_maps:  [B, T, N]   — spatial artifact heatmap (for visualization)
        """
        B, T, N, D = patch_tokens.shape
        
        # Process each frame independently
        x = rearrange(patch_tokens, 'b t n d -> (b t) n d')
        
        # Self-attention: each patch attends to all others in same frame
        # Inconsistent patches will have high attention residuals
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.spatial_norm(x + attn_out)              # [(B*T), N, D]
        
        # Score each patch for artifact likelihood
        artifact_scores = self.artifact_scorer(x)        # [(B*T), N, 1]
        artifact_scores = rearrange(artifact_scores, '(b t) n 1 -> b t n', b=B, t=T)
        
        # Weighted aggregation: suspicious patches get higher weight
        weights = rearrange(artifact_scores, 'b t n -> (b t) n 1')  # soft attention
        x_weighted = (x * weights).sum(dim=1)            # [(B*T), D]
        frame_feats = self.patch_aggregator(x_weighted)
        frame_feats = rearrange(frame_feats, '(b t) d -> b t d', b=B, t=T)
        
        return frame_feats, artifact_scores


class TemporalInconsistencyModule(nn.Module):
    """
    Detect temporal inconsistencies between frames at patch level.
    
    Key insight: Deepfakes often have frame-to-frame inconsistencies
    that don't exist in real videos (flickering artifacts, temporal
    incoherence in specific face regions).
    """
    def __init__(self, feature_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Cross-frame attention: each patch in frame t attends to frame t+1
        self.cross_frame_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
        # Measure inconsistency between consecutive frames
        self.inconsistency_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )
        
        self.temporal_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: [B, T, N, D]
        Returns:
            temporal_feat: [B, D]
        """
        B, T, N, D = patch_tokens.shape
        inconsistencies = []
        
        for t in range(T - 1):
            curr = patch_tokens[:, t]       # [B, N, D]
            next_ = patch_tokens[:, t + 1]  # [B, N, D]
            
            # Cross-attention: current frame patches query next frame patches
            # High residual = high temporal inconsistency
            attn_out, _ = self.cross_frame_attn(curr, next_, next_)
            residual = curr - attn_out       # [B, N, D] — difference signal
            
            # Concatenate original + residual, project
            combined = torch.cat([curr, residual], dim=-1)  # [B, N, 2D]
            incons_feat = self.inconsistency_proj(combined) # [B, N, D]
            
            # Pool over patches
            incons_feat = incons_feat.mean(dim=1)           # [B, D]
            inconsistencies.append(incons_feat)
        
        # Stack temporal inconsistencies: [B, T-1, D]
        incons_seq = torch.stack(inconsistencies, dim=1)
        
        # Pool over temporal dimension
        temporal_feat = self.temporal_pool(incons_seq.mean(dim=1))  # [B, D]
        return temporal_feat


class SpatialFrequencyModule(nn.Module):
    """
    Capture frequency-domain artifacts that are often present in deepfakes.
    
    Instead of DCT (complex), we use learned frequency filters applied
    to the patch feature space — effectively learned spectral analysis.
    """
    def __init__(self, feature_dim: int = 384, num_freq_filters: int = 64):
        super().__init__()
        
        # Learned frequency decomposition
        # Projects features into frequency-like components
        self.freq_encoder = nn.Sequential(
            nn.Linear(feature_dim, num_freq_filters * 2),
            nn.GELU(),
            nn.Linear(num_freq_filters * 2, num_freq_filters)
        )
        
        # Aggregate frequency patterns across spatial positions
        self.freq_aggregator = nn.Sequential(
            nn.Linear(num_freq_filters, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2)
        )
        
        self.output_dim = feature_dim // 2
    
    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: [B, T, N, D]
        Returns:
            freq_feat: [B, feature_dim // 2]
        """
        B, T, N, D = patch_tokens.shape
        
        # Reshape for processing
        x = rearrange(patch_tokens, 'b t n d -> (b t n) d')
        
        # Encode into frequency components
        freq = self.freq_encoder(x)                     # [(B*T*N), F]
        freq = rearrange(freq, '(b t n) f -> b t n f', b=B, t=T, n=N)
        
        # Statistics over spatial positions capture frequency distribution
        freq_mean = freq.mean(dim=2)                    # [B, T, F]
        freq_std = freq.std(dim=2)                      # [B, T, F]
        freq_feat = torch.cat([freq_mean, freq_std], dim=-1).mean(dim=1)  # [B, 2F]
        
        return self.freq_aggregator(freq_feat)           # [B, D//2]


# ============================================================
#  3. LIGHTWEIGHT GNN (replaces heavy MinCut approach)
# ============================================================

class LightweightSemanticGNN(nn.Module):
    """
    Lightweight GNN that:
    1. Uses semantic (feature-based) KNN instead of spatial KNN
    2. Processes frames independently then aggregates temporally
    3. Avoids dense adjacency matrices entirely
    """
    def __init__(
        self,
        feature_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        K: int = 8,
        T: int = 8,
        N: int = 256
    ):
        super().__init__()
        self.K = K
        self.T = T
        self.N = N
        
        head_dim = feature_dim // num_heads
        
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=feature_dim,
                out_channels=head_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        
        # Precompute temporal edges (static — same-patch across frames)
        self.register_buffer(
            'temporal_edges',
            build_temporal_edges(T, N, device=torch.device('cpu'))
        )
        
        # Frame-level aggregation after GNN
        self.frame_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )
        
        # Temporal aggregation over frames
        self.temporal_agg = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: [B, T, N, D]
        Returns:
            graph_feat: [B, D]
        """
        B, T, N, D = patch_tokens.shape
        graph_feats = []
        
        for b in range(B):
            # Build semantic edges for this sample
            # Per-frame semantic KNN edges
            frame_edges_list = []
            for t in range(T):
                frame_patches = patch_tokens[b, t]          # [N, D]
                sem_edges = build_semantic_knn_edges(frame_patches, K=self.K)
                offset = t * N
                frame_edges_list.append(sem_edges + offset)
            
            spatial_edges = torch.cat(frame_edges_list, dim=1)    # [2, T*N*E]
            temporal_edges = self.temporal_edges.to(patch_tokens.device)
            
            all_edges = torch.cat([spatial_edges, temporal_edges], dim=1)
            
            x = rearrange(patch_tokens[b], 't n d -> (t n) d')   # [T*N, D]
            
            # GNN layers with residual connections
            for gat, norm in zip(self.gat_layers, self.layer_norms):
                x_new = gat(x, all_edges)
                x_new = F.gelu(x_new)
                x = norm(x + x_new)                               # residual
            
            # Reshape back to [T, N, D]
            x = rearrange(x, '(t n) d -> t n d', t=T, n=N)
            
            # Pool patches within each frame
            frame_repr = self.frame_pool(x.mean(dim=1))           # [T, D]
            graph_feats.append(frame_repr)
        
        # Stack: [B, T, D]
        graph_feats = torch.stack(graph_feats, dim=0)
        
        # Temporal aggregation over frames
        out, _ = self.temporal_agg(graph_feats)
        return out.mean(dim=1)                                     # [B, D]


# ============================================================
#  4. FEATURE EXTRACTOR (unchanged interface, minor addition)
# ============================================================

class VideoFeatureExtractor(nn.Module):
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, weight_path=None):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        
        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            vit_state_dict = {
                k.replace("vit.", "", 1): v
                for k, v in state_dict.items() if k.startswith("vit.")
            }
            self.vit.load_state_dict(vit_state_dict, strict=False)
        
        # Freeze all
        for p in self.vit.parameters():
            p.requires_grad = False
        
        # Unfreeze last 2 blocks + all LayerNorms (for fine-tuning)
        for block in self.vit.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        for module in self.vit.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
        
        self.D = feature_dim
    
    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.vit.forward_features(x_flat)
        cls_tokens   = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']
        return {
            'cls':   cls_tokens.reshape(B, T, self.D),
            'patch': patch_tokens.reshape(B, T, 256, self.D)   # NOTE: now [B,T,N,D]
        }


# ============================================================
#  5. IMPROVED FUSED MODEL
# ============================================================

class ImprovedFusedModel(nn.Module):
    """
    Three complementary streams, each targeting different artifact types:
    
    Stream 1 — BiLSTM on CLS tokens:
        Global temporal dynamics, overall face consistency over time.
    
    Stream 2 — Patch Artifact Detector + Temporal Inconsistency:
        Local spatial artifacts, frame-to-frame inconsistencies.
        Directly exploits patch tokens without heavy graph overhead.
    
    Stream 3 — Lightweight Semantic GNN:
        Semantic region relationships, structured manipulation patterns.
        Uses feature-similarity edges instead of spatial grid edges.
    
    All three are projected to the same dimension before fusion,
    then fused with learned attention weights (not naive concatenation).
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        num_gcn_layers: int = 2,
        num_heads: int = 6,
        mlp_dim: int = 512,
        fusion_dim: int = 256,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim
        
        # ── Shared ViT backbone ───────────────────────────────────────────────
        self.vit = VideoFeatureExtractor(
            vit_name=vit_name,
            feature_dim=feature_dim,
            weight_path=vit_weight_path
        )
        
        # ── Stream 1: BiLSTM on CLS tokens ───────────────────────────────────
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_of_frames, feature_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.bilstm_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ── Stream 2a: Patch Artifact Detection ───────────────────────────────
        self.patch_detector = PatchArtifactDetector(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.patch_bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ── Stream 2b: Temporal Inconsistency ────────────────────────────────
        self.temporal_incons = TemporalInconsistencyModule(
            feature_dim=feature_dim,
            num_heads=num_heads
        )
        self.incons_proj = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ── Stream 3: Frequency artifacts ────────────────────────────────────
        self.freq_module = SpatialFrequencyModule(
            feature_dim=feature_dim,
            num_freq_filters=64
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(feature_dim // 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # ── Attention-based Fusion ────────────────────────────────────────────
        # Each stream gets a scalar attention weight — learned, not fixed
        self.num_streams = 4
        self.stream_attention = nn.Sequential(
            nn.Linear(fusion_dim * self.num_streams, self.num_streams),
            nn.Softmax(dim=-1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        cls_features   = feats['cls']           # [B, T, D]
        patch_features = feats['patch']         # [B, T, N, D]
        
        # ── Stream 1: BiLSTM on CLS ──────────────────────────────────────────
        cls_in = cls_features + self.temporal_pos_embed
        lstm_out, _ = self.bilstm(cls_in)
        s1 = self.bilstm_proj(lstm_out.mean(dim=1))    # [B, fusion_dim]
        
        # ── Stream 2a: Patch artifact detection ──────────────────────────────
        frame_feats, artifact_maps = self.patch_detector(patch_features)
        patch_out, _ = self.patch_bilstm(frame_feats)
        s2a = self.patch_proj(patch_out.mean(dim=1))   # [B, fusion_dim]
        
        # ── Stream 2b: Temporal inconsistency ────────────────────────────────
        s2b = self.incons_proj(
            self.temporal_incons(patch_features)
        )                                              # [B, fusion_dim]
        
        # ── Stream 3: Frequency artifacts ────────────────────────────────────
        s3 = self.freq_proj(
            self.freq_module(patch_features)
        )                                              # [B, fusion_dim]
        
        # ── Attention fusion ─────────────────────────────────────────────────
        streams = torch.stack([s1, s2a, s2b, s3], dim=1)        # [B, 4, fusion_dim]
        concat_streams = streams.reshape(B, -1)                  # [B, 4*fusion_dim]
        attn_weights = self.stream_attention(concat_streams)     # [B, 4]
        attn_weights = attn_weights.unsqueeze(-1)                # [B, 4, 1]
        
        fused = (streams * attn_weights).sum(dim=1)              # [B, fusion_dim]
        logits = self.classifier(fused)
        
        return logits, artifact_maps