import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchinfo import summary
from zahin_model_video import MyModel

from create_graph import create_frame_graph  # if you keep create_frame_graph in another file, update this import


# ====================== Logging Setup ======================
def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("=== Training Started ===")


# ====================== Frame Sampling ======================
def sample_frames(frame_paths, num_frames):
    """
    Sample num_frames from a video folder at equal intervals.
    If the folder has fewer frames than requested, indices may repeat.
    """
    frame_paths = sorted(frame_paths)
    total = len(frame_paths)
    if total == 0:
        raise ValueError("Empty frame folder found.")

    if total == num_frames:
        return frame_paths

    indices = np.linspace(0, total - 1, num_frames)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total - 1)
    return [frame_paths[i] for i in indices]


# ====================== Video Discovery + Split ======================
def get_video_groups(root):
    """
    Returns grouped video folders by subtype so that splitting is done
    separately inside each subtype, avoiding leakage.

    Example groups:
      manipulated_sequences/deepfakedetection/frames/<video_id>/
      manipulated_sequences/neuraltexture/frames/<video_id>/
      original_sequences/actors/frames/<video_id>/
      original_sequences/youtube/frames/<video_id>/
    """
    root = Path(root)
    groups = defaultdict(list)

    candidates = []
    candidates += list(root.glob("manipulated_sequences/*/c23/frames/*"))
    candidates += list(root.glob("original_sequences/*/c23/frames/*"))
    # print(len(candidates))

    for video_dir in candidates:
        if not video_dir.is_dir():
            continue

        # subtype name: deepfakedetection / neuraltexture / actors / youtube / etc.
        subtype = video_dir.parent.parent.name

        # label from the top-level folder
        label = 1 if "manipulated_sequences" in str(video_dir) else 0

        frame_paths = sorted(video_dir.glob("*.png"))
        if len(frame_paths) == 0:
            continue

        groups[(label, subtype)].append((video_dir, frame_paths))

    return groups


def split_train_val_by_group(root, num_frames, val_ratio=0.2, seed=42):
    """
    Split each subtype separately with 80/20 ratio.
    This prevents leakage across train/val.
    """
    rng = random.Random(seed)
    groups = get_video_groups(root)

    train_samples = []
    val_samples = []

    for (label, subtype), items in groups.items():
        items = list(items)
        rng.shuffle(items)

        n_total = len(items)
        n_val = max(1, int(round(n_total * val_ratio))) if n_total > 1 else 0
        n_train = n_total - n_val

        train_items = items[:n_train]
        val_items = items[n_train:]

        for video_dir, frame_paths in train_items:
            selected = sample_frames(frame_paths, num_frames)
            train_samples.append((selected, label))

        for video_dir, frame_paths in val_items:
            selected = sample_frames(frame_paths, num_frames)
            val_samples.append((selected, label))

        logging.info(
            f"Subtype={subtype:20s} | Total={n_total:5d} | Train={len(train_items):5d} | Val={len(val_items):5d}"
        )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


# ====================== Dataset ======================
class FFPPVideoDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def load_item(self, sample):
        frame_paths, label = sample

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img_t = self.transform(img) if self.transform else img
            frames.append(img_t)

        video = torch.stack(frames, dim=0)  # [T, 3, H, W]
        label = torch.tensor(label, dtype=torch.long)
        return video, label

    def __getitem__(self, index):
        return self.load_item(self.samples[index])


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training on FF++")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to FF++ root folder")
    parser.add_argument("--vit-name", type=str, default="dinov2_vits14")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames sampled per video")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=8)
    args = parser.parse_args()

    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Current device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----------------- Data splitting -----------------
    train_samples, val_samples = split_train_val_by_group(
        root=args.data_root,
        num_frames=args.num_frames,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    logging.info(f"Total train videos: {len(train_samples)}")
    logging.info(f"Total val videos:   {len(val_samples)}")

    train_dataset = FFPPVideoDataset(train_samples, transform=get_transforms())
    val_dataset = FFPPVideoDataset(val_samples, transform=get_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )


    # ----------------- Model -----------------
    model = MyModel(vit_name=args.vit_name).to(device)

    if torch.cuda.is_available():
        summary(model, input_size=(1, args.num_frames, 3, 224, 224), device="cuda")

    # Create graph once for the fixed number of frames
    edge_index = create_frame_graph(T=args.num_frames, N=256, K=args.knn_k).to(device)

    # ----------------- Optimizer, Loss, Scheduler -----------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=0.001,
        min_lr=1e-6
    )

    # ----------------- Checkpointing -----------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pth"
    best_model_path = Path(args.checkpoint_dir) / "best_model.pth"

    start_epoch = 0
    best_auc = 0.0

    if args.resume and checkpoint_path.exists():
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        logging.info(f"Resumed from epoch {start_epoch} | Best AUC so far: {best_auc:.4f}")

    # ====================== Training Loop ======================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)   # [B, T, 3, H, W]
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(videos, edge_index)              # IMPORTANT: pass edge_index
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_train_preds.extend(probs)
            all_train_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.inference_mode():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(videos, edge_index)
                loss = criterion(logits, labels)

                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_val_preds.extend(probs)
                all_val_labels.extend(labels.detach().cpu().numpy())

        # ----------------- Metrics -----------------
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        train_pred_bin = (np.array(all_train_preds) > 0.5).astype(int)
        val_pred_bin = (np.array(all_val_preds) > 0.5).astype(int)

        train_acc = accuracy_score(all_train_labels, train_pred_bin)
        val_acc = accuracy_score(all_val_labels, val_pred_bin)

        try:
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
        except ValueError:
            train_auc = 0.5

        try:
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except ValueError:
            val_auc = 0.5

        logging.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
        )

        # ----------------- Scheduler -----------------
        scheduler.step(val_auc)

        # ----------------- Checkpointing -----------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "args": vars(args),
        }, checkpoint_path)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "args": vars(args),
            }, best_model_path)
            logging.info(f"New best model saved! Val AUC = {val_auc:.4f}")

    logging.info("=== Training Finished ===")
    logging.info(f"Best Validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()