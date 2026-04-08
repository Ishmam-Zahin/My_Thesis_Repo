import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from torchinfo import summary

from zahin_model import My_Model   # your model file


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


# ====================== Dataset (unchanged) ======================
class My_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def load_item(self, path):
        path_str = str(path)
        img = Image.open(path).convert('RGB')
        img_t = self.transform(img) if self.transform else img
        label = 1 if ('fake' in path_str or 'manipulated_sequences' in path_str) else 0
        return img_t, label

    def __getitem__(self, index):
        return self.load_item(self.paths[index])


def load_paths(root):
    return list(Path(root).glob('**/frames/*/*.png'))


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # DINOv2 normalization (recommended)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ====================== Main ======================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"----------------- Current device: {device} ---------------------------")

    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path',   type=str, required=True)
    parser.add_argument('--vit-name',   type=str, default="dinov2_vits14")
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints")
    parser.add_argument('--resume',     action='store_true', help="Resume from latest checkpoint")
    args = parser.parse_args()

    setup_logging()

    # ----------------- Data -----------------
    train_paths = load_paths(args.train_path)
    val_paths   = load_paths(args.val_path)

    train_dataset = My_Dataset(train_paths, transform=get_transforms())
    val_dataset   = My_Dataset(val_paths,   transform=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ----------------- Model -----------------
    model = My_Model(vit_name=args.vit_name).to(device)

    # Print model summary once
    if torch.cuda.is_available():
        summary(model, input_size=(1, 3, 224, 224), device="cuda")

    # ----------------- Optimizer, Loss, Scheduler -----------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
        threshold=0.001, min_lr=1e-6
    )

    # ----------------- Checkpointing -----------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pth"
    best_model_path = Path(args.checkpoint_dir) / "best_model.pth"

    start_epoch = 0
    best_auc = 0.0

    # Resume from latest checkpoint
    if args.resume and checkpoint_path.exists():
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('best_auc', 0.0)
        logging.info(f"Resumed training from epoch {start_epoch} | Best AUC so far: {best_auc:.4f}")

    # ====================== Training Loop ======================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)                    # (B, 2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_train_preds.extend(probs)
            all_train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.inference_mode():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_val_preds.extend(probs)
                all_val_labels.extend(labels.cpu().numpy())

        # ----------------- Metrics -----------------
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        train_acc = accuracy_score(all_train_labels, np.array(all_train_preds) > 0.5)
        val_acc   = accuracy_score(all_val_labels,   np.array(all_val_preds)   > 0.5)

        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        val_auc   = roc_auc_score(all_val_labels,   all_val_preds)

        logging.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
        )

        # ----------------- Scheduler (on Val AUC) -----------------
        scheduler.step(val_auc)

        # ----------------- Checkpointing -----------------
        # Save latest checkpoint (can resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_auc': best_auc,
        }, checkpoint_path)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
            }, best_model_path)
            logging.info(f"✅ New best model saved! Val AUC = {val_auc:.4f}")

    logging.info("=== Training Finished ===")
    logging.info(f"Best Validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()