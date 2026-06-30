"""
graph_diagnostic.py

Run this BEFORE full training to check whether tau_s / tau_t are well-calibrated
for your DINOv2 patch features.

Usage:
    python graph_diagnostic.py --config train.yaml --num_samples 10

What it reports per sample:
  - consistency edge count + density  (want > 5% of max possible)
  - inconsistency edge count + density
  - temporal edge count (subset of consistency)
  - spatial edge count  (subset of consistency)
  - recommendation: raise / lower / keep tau_s and tau_t
"""

import argparse
import importlib.util
import random
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from helpers.dataset_loader import get_dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model_class(model_path: str, class_name: str):
    model_path = Path(model_path).resolve()
    spec = importlib.util.spec_from_file_location("dynamic_model_module", str(model_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def edge_density(edge_count: int, num_nodes: int) -> float:
    """Fraction of all possible directed edges that exist."""
    max_edges = num_nodes * (num_nodes - 1)
    return edge_count / max_edges if max_edges > 0 else 0.0


def bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value*100:.1f}%"


# ── main diagnostic ───────────────────────────────────────────────────────────

def run_diagnostic(config_path: str, num_samples: int = 10):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    T = config["training"]["num_frames"]
    N = 256          # DINOv2-vits14 → 16×16 = 256 patches per frame
    num_nodes = T * N

    # Max possible directed edges (excluding self-loops)
    max_directed = num_nodes * (num_nodes - 1)
    # Max possible within-frame spatial edges (one frame)
    max_spatial_per_frame = N * (N - 1)
    # Max possible temporal edges (same-patch, consecutive frames)
    max_temporal = (T - 1) * N * 2   # bidirectional

    print("\n" + "=" * 70)
    print("  GRAPH EDGE DIAGNOSTIC")
    print(f"  T={T} frames × N={N} patches → {num_nodes} nodes per sample")
    print(f"  tau_s = {model_config.get('tau_s', 0.6)}")
    print(f"  tau_t = {model_config.get('tau_t', 0.6)}")
    print("=" * 70)

    # ── load model (graph construction only, skip ViT forward) ──────────────
    ModelClass = load_model_class(model_config["model_path"], model_config["model_class"])
    model = ModelClass(
        vit_name=model_config["vit_name"],
        feature_dim=384,
        dropout=model_config.get("dropout", 0.2),
        num_of_frames=T,
        num_gcn_layers=model_config.get("num_gcn_layers", 2),
        num_clusters=model_config.get("num_clusters", 512),
        num_transformer_blocks=model_config.get("num_transformer_blocks", 1),
        num_heads=model_config.get("num_heads", 8),
        mlp_dim=model_config.get("mlp_dim", 512),
        tau_s=model_config.get("tau_s", 0.6),
        tau_t=model_config.get("tau_t", 0.6),
        graph_eps=model_config.get("graph_eps", 1e-4),
        local_window=model_config.get("local_window", 3),
        vit_weight_path=model_config.get("vit_weight"),
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ── dataset ─────────────────────────────────────────────────────────────
    transform = get_transforms()
    train_dataset, _ = get_dataset(
        config["data"]["json_root"],
        config["data"]["dataset_name"],
        transform,
    )

    indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)

    # Accumulators
    cons_densities, incon_densities = [], []
    temporal_densities, spatial_densities = [], []
    warnings = []

    print(f"\nAnalysing {len(subset)} samples...\n")

    for sample_idx, (video, label) in enumerate(loader):
        video = video.to(device)                         # [1, T, C, H, W]

        with torch.no_grad():
            feats = model.vit(video)
            patch_tokens_flat = feats["patch"][0]        # [T*N, D]

        T_ = model.num_of_frames
        patch_tokens = patch_tokens_flat.view(T_, N, model.feature_dim)

        edge_index_cons, edge_attr_cons, edge_index_incon, edge_attr_incon = \
            model._build_graphs_for_sample(patch_tokens)

        n_cons  = edge_index_cons.shape[1]
        n_incon = edge_index_incon.shape[1]

        # ── split consistency into spatial vs temporal ───────────────────────
        # Spatial edges: both endpoints in the same frame
        # (src // N == dst // N)
        src_c, dst_c = edge_index_cons[0], edge_index_cons[1]
        spatial_mask  = (src_c // N) == (dst_c // N)
        temporal_mask = ~spatial_mask

        n_spatial  = spatial_mask.sum().item()
        n_temporal = temporal_mask.sum().item()

        # ── densities ───────────────────────────────────────────────────────
        d_cons    = n_cons  / max_directed
        d_incon   = n_incon / max_directed
        d_spatial = n_spatial  / (max_spatial_per_frame * T_) if (max_spatial_per_frame * T_) > 0 else 0
        d_temporal= n_temporal / max_temporal if max_temporal > 0 else 0

        cons_densities.append(d_cons)
        incon_densities.append(d_incon)
        temporal_densities.append(d_temporal)
        spatial_densities.append(d_spatial)

        label_str = "real" if label.item() == 0 else "fake"
        print(f"  Sample {sample_idx+1:2d}  [{label_str}]")
        print(f"    Consistency  edges: {n_cons:7,d}  density {bar(d_cons)}")
        print(f"      ↳ spatial  edges: {n_spatial:7,d}  density {bar(d_spatial)}  (within-frame)")
        print(f"      ↳ temporal edges: {n_temporal:7,d}  density {bar(d_temporal)}  (cross-frame, same patch)")
        print(f"    Inconsistency edges:{n_incon:7,d}  density {bar(d_incon)}")

        # Per-sample warnings
        if d_cons < 0.05:
            warnings.append(f"  ⚠  Sample {sample_idx+1}: consistency density {d_cons*100:.1f}% < 5% → tau_s likely too high")
        if d_temporal < 0.10:
            warnings.append(f"  ⚠  Sample {sample_idx+1}: temporal density {d_temporal*100:.1f}% < 10% → tau_t likely too high")
        print()

    # ── aggregate summary ────────────────────────────────────────────────────
    avg_cons     = np.mean(cons_densities)
    avg_incon    = np.mean(incon_densities)
    avg_spatial  = np.mean(spatial_densities)
    avg_temporal = np.mean(temporal_densities)

    print("=" * 70)
    print("  SUMMARY (averages across all samples)")
    print("=" * 70)
    print(f"  Consistency  density (avg): {bar(avg_cons)}")
    print(f"    ↳ spatial  density (avg): {bar(avg_spatial)}")
    print(f"    ↳ temporal density (avg): {bar(avg_temporal)}")
    print(f"  Inconsistency density (avg):{bar(avg_incon)}")

    # ── recommendations ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    tau_s = model_config.get("tau_s", 0.6)
    tau_t = model_config.get("tau_t", 0.6)

    # tau_s recommendation
    if avg_cons < 0.02:
        rec_tau_s = max(0.2, tau_s - 0.2)
        print(f"  tau_s: MUCH TOO HIGH  → lower from {tau_s} to ~{rec_tau_s:.1f}")
        print(f"         Consistency graph is nearly empty ({avg_cons*100:.1f}%).")
        print(f"         GAT has almost nothing to aggregate spatially.")
    elif avg_cons < 0.05:
        rec_tau_s = max(0.2, tau_s - 0.1)
        print(f"  tau_s: TOO HIGH       → lower from {tau_s} to ~{rec_tau_s:.1f}")
        print(f"         Consistency graph is sparse ({avg_cons*100:.1f}% < 5% target).")
    elif avg_cons > 0.40:
        rec_tau_s = min(0.9, tau_s + 0.1)
        print(f"  tau_s: TOO LOW        → raise from {tau_s} to ~{rec_tau_s:.1f}")
        print(f"         Consistency graph is very dense ({avg_cons*100:.1f}% > 40%).")
        print(f"         Nearly all patches connected; threshold not selective enough.")
    else:
        print(f"  tau_s: OK             → keep at {tau_s}  ({avg_cons*100:.1f}% density, target 5–40%)")

    # tau_t recommendation
    if avg_temporal < 0.10:
        rec_tau_t = max(0.2, tau_t - 0.1)
        print(f"  tau_t: TOO HIGH       → lower from {tau_t} to ~{rec_tau_t:.1f}")
        print(f"         Temporal edges are sparse ({avg_temporal*100:.1f}% < 10% target).")
        print(f"         Model can't learn cross-frame consistency well.")
    elif avg_temporal > 0.80:
        rec_tau_t = min(0.9, tau_t + 0.1)
        print(f"  tau_t: TOO LOW        → raise from {tau_t} to ~{rec_tau_t:.1f}")
        print(f"         Almost every patch has temporal edges ({avg_temporal*100:.1f}% > 80%).")
        print(f"         Threshold not filtering anything; try raising it.")
    else:
        print(f"  tau_t: OK             → keep at {tau_t}  ({avg_temporal*100:.1f}% density, target 10–80%)")

    # Per-sample warnings
    if warnings:
        print()
        for w in warnings:
            print(w)

    print("\n" + "=" * 70)
    print("  RAW NUMBERS")
    print("=" * 70)
    print(f"  Max possible directed edges (T*N nodes):  {max_directed:,}")
    print(f"  Max possible spatial edges  (T frames):   {max_spatial_per_frame * T_:,}")
    print(f"  Max possible temporal edges (bidirec.):   {max_temporal:,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True, help="Path to train.yaml")
    parser.add_argument("--num_samples", type=int, default=10,    help="Number of samples to diagnose")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    run_diagnostic(args.config, args.num_samples)