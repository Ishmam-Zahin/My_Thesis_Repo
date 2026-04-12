# Full train.py with Early Stopping + vit-name + original features restored

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

from create_graph import create_frame_graph


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
    root = Path(root)
    groups = defaultdict(list)

    candidates = []
    candidates += list(root.glob("manipulated_sequences/*/c23/frames/*"))
    candidates += list(root.glob("original_sequences/*/c23/frames/*"))

    for video_dir in candidates:
        if not video_dir.is_dir():
            continue

        subtype = video_dir.parent.parent.name
        label = 1 if "manipulated_sequences" in str(video_dir) else 0

        frame_paths = sorted(video_dir.glob("*.png"))
        if len(frame_paths) == 0:
            continue

        groups[(label, subtype)].append((video_dir, frame_paths))

    return groups


def split_train_val_by_group(root, num_frames, val_ratio=0.2, seed=42):
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
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img_t = self.transform(img) if self.transform else img
            frames.append(img_t)

        video = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return video, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training")

    # Original args
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--vit-name", type=str, default="dinov2_vits14")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=8)

    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-delta", type=float, default=1e-3)

    args = parser.parse_args()

    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Current device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----------------- Data -----------------
    train_samples, val_samples = split_train_val_by_group(
        args.data_root, args.num_frames, args.val_ratio, args.seed
    )

    logging.info(f"Total train videos: {len(train_samples)}")
    logging.info(f"Total val videos:   {len(val_samples)}")

    train_loader = DataLoader(
        FFPPVideoDataset(train_samples, get_transforms()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        FFPPVideoDataset(val_samples, get_transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # ----------------- Model -----------------
    model = MyModel(vit_name=args.vit_name).to(device)

    edge_index = create_frame_graph(
        T=args.num_frames,
        N=256,
        K=args.knn_k
    ).to(device)

    if torch.cuda.is_available():
        summary(
            model,
            input_data=(torch.randn(1, args.num_frames, 3, 224, 224).to(device), edge_index),
            device="cuda"
        )

    # ----------------- Optimizer -----------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # ----------------- Checkpoint -----------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = Path(args.checkpoint_dir) / "best_model.pth"

    best_val_auc = 0.0
    epochs_no_improve = 0

    # ====================== Training ======================
    for epoch in range(args.epochs):
        model.train()
        train_preds, train_labels = [], []

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(videos, edge_index)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            train_preds.extend(probs)
            train_labels.extend(labels.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_preds)

        # ----------------- Validation -----------------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                videos = videos.to(device)
                labels = labels.to(device)

                logits = model(videos, edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        logging.info(
            f"Epoch {epoch+1} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
        )

        scheduler.step(val_auc)

        # ----------------- Early Stopping -----------------
        if val_auc > best_val_auc + args.early_stop_delta:
            best_val_auc = val_auc
            epochs_no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "args": vars(args),
            }, best_model_path)

            logging.info(f"New best model saved! Val AUC = {val_auc:.4f}")

        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= args.early_stop_patience:
                logging.info("Early stopping triggered!")
                break

    logging.info("=== Training Finished ===")
    logging.info(f"Best Validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
