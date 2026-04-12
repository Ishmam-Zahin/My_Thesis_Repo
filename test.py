import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from zahin_model_video import MyModel
from create_graph import create_frame_graph


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REAL_KEYWORDS = {"real", "original", "bonafide", "genuine", "authentic", "Celeb-real", "YouTube-real"}
FAKE_KEYWORDS = {"fake", "faked", "manipulated", "synthesis", "synthesized", "deepfake", "spoof", "Celeb-synthesis"}


# -------------------------- Logging --------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )


# -------------------------- Utils --------------------------
def get_transforms():
    # Matches train.py exactly
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def sample_frames(frame_paths: Sequence[Path], num_frames: int) -> List[Path]:
    frame_paths = sorted(frame_paths)
    total = len(frame_paths)
    if total == 0:
        raise ValueError("Empty frame folder found.")
    if total == num_frames:
        return list(frame_paths)

    indices = np.linspace(0, total - 1, num_frames)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total - 1)
    return [frame_paths[i] for i in indices]


def infer_label_from_path(path: Path, root: Path) -> Optional[int]:
    """
    Heuristic label inference for common deepfake dataset layouts.

    Returns:
        0 for real, 1 for fake, or None if the label cannot be inferred.
    """
    rel_parts = [p.lower() for p in path.relative_to(root).parts]

    # Look from leaf -> root so the nearest label-like folder wins.
    for part in reversed(rel_parts):
        tokens = set(part.replace("-", "_").replace(" ", "_").split("_"))
        if tokens & REAL_KEYWORDS:
            return 0
        if tokens & FAKE_KEYWORDS:
            return 1

    # Fallback to filename/path substring checks.
    joined = "/".join(rel_parts)
    if any(k in joined for k in REAL_KEYWORDS):
        return 0
    if any(k in joined for k in FAKE_KEYWORDS):
        return 1

    return None


def discover_video_folders(root: Path) -> List[Path]:
    """
    Recursively find folders that directly contain frame images.
    Each folder is treated as one video.
    """
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)
        frames = [dirpath / f for f in filenames if is_image_file(Path(f))]
        if frames:
            candidates.append(dirpath)

    # Remove duplicates and keep stable ordering.
    unique = []
    seen = set()
    for p in sorted(candidates):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build_samples(root: str, num_frames: int) -> List[Tuple[List[Path], int]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    video_dirs = discover_video_folders(root_path)
    samples = []
    unresolved = []

    for vdir in video_dirs:
        frame_paths = sorted([p for p in vdir.iterdir() if p.is_file() and is_image_file(p)])
        if not frame_paths:
            continue

        label = infer_label_from_path(vdir, root_path)
        if label is None:
            unresolved.append(str(vdir))
            continue

        selected = sample_frames(frame_paths, num_frames)
        samples.append((selected, label))

    if unresolved and not samples:
        raise RuntimeError(
            "Could not infer labels from the dataset folder structure. "
            "Please place videos under folders containing real/fake style names. "
            f"Example unresolved folder: {unresolved[0]}"
        )

    if unresolved:
        logging.warning(
            f"Skipped {len(unresolved)} folders because labels could not be inferred. "
            f"Example: {unresolved[0]}"
        )

    return samples


# -------------------------- Dataset --------------------------
class FrameFolderDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img) if self.transform else img
            frames.append(img)

        video = torch.stack(frames, dim=0)  # [T, 3, H, W]
        label = torch.tensor(label, dtype=torch.long)
        return video, label


# -------------------------- Checkpoint --------------------------
def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    return ckpt


# -------------------------- Evaluation --------------------------
@torch.inference_mode()
def evaluate_dataset(model, loader, edge_index, device, dataset_name: str):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for videos, labels in tqdm(loader, desc=f"Testing {dataset_name}"):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(videos, edge_index)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    preds = (np.array(all_probs) >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds) if len(all_labels) else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, preds, labels=[0, 1]) if len(all_labels) else np.zeros((2, 2), dtype=int)

    print(f"\n=== {dataset_name} ===")
    print(f"Samples: {len(all_labels)}")
    print(f"Avg loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}" if not np.isnan(auc) else "AUC:      N/A (only one class present)")
    print("Confusion matrix [real, fake]:")
    print(cm)
    if len(set(all_labels)) > 1:
        print("Classification report:")
        print(classification_report(all_labels, preds, target_names=["real", "fake"], digits=4))

    return {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "num_samples": len(all_labels),
        "confusion_matrix": cm,
    }


# -------------------------- Main --------------------------
def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Test MyModel on Celeb-DF v1, Celeb-DF v2, and UADFV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--celeb1-root", type=str, default=None, help="Path to Celeb-DF v1 root folder")
    parser.add_argument("--celeb2-root", type=str, default=None, help="Path to Celeb-DF v2 root folder")
    parser.add_argument("--uadfv-root", type=str, default=None, help="Path to UADFV root folder")
    parser.add_argument("--vit-name", type=str, default="dinov2_vits14")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames sampled per video")
    parser.add_argument("--knn-k", type=int, default=8, help="K in create_frame_graph")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or leave empty for auto")
    args = parser.parse_args()

    provided = {
        "Celeb-DF v1": args.celeb1_root,
        "Celeb-DF v2": args.celeb2_root,
        "UADFV": args.uadfv_root,
    }
    provided = {k: v for k, v in provided.items() if v}

    if not provided:
        raise ValueError("Provide at least one dataset root: --celeb1-root, --celeb2-root, or --uadfv-root")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # Model input check from train.py:
    # - videos are [B, T, 3, 224, 224]
    # - T is fixed by --num-frames
    # - each frame is resized to 224x224 and normalized
    model = MyModel(vit_name=args.vit_name).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    edge_index = create_frame_graph(T=args.num_frames, N=256, K=args.knn_k).to(device)

    # Print a small shape sanity check.
    logging.info(f"Expected model input: [B, {args.num_frames}, 3, 224, 224]")
    logging.info(f"Graph edge_index shape: {tuple(edge_index.shape)}")

    transform = get_transforms()
    results = {}

    for dataset_name, root in provided.items():
        logging.info(f"Discovering samples for {dataset_name}: {root}")
        samples = build_samples(root, args.num_frames)
        if not samples:
            logging.warning(f"No valid samples found for {dataset_name}.")
            continue

        labels = [label for _, label in samples]
        real_count = sum(1 for x in labels if x == 0)
        fake_count = sum(1 for x in labels if x == 1)
        logging.info(
            f"{dataset_name} -> videos: {len(samples)} | real: {real_count} | fake: {fake_count}"
        )

        dataset = FrameFolderDataset(samples, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Verify one sample shape against the model expectation.
        sample_video, sample_label = dataset[0]
        logging.info(
            f"Sample shape for {dataset_name}: video={tuple(sample_video.shape)} label={int(sample_label)}"
        )

        results[dataset_name] = evaluate_dataset(model, loader, edge_index, device, dataset_name)

    print("\n=== Summary ===")
    for name, res in results.items():
        print(f"{name}: acc={res['acc']:.4f}, auc={res['auc']:.4f}, samples={res['num_samples']}")


if __name__ == "__main__":
    main()
