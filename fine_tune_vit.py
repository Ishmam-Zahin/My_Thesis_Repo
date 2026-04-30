import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("train", []), data.get("test", [])


def determine_labels(path):
    real_words = ["real", "original"]
    fake_words = ["fake", "manipulated", "synthesis"]
    path_str = str(path).lower()

    if "dfdc" in path_str:
        # Expected format: ...+<label>...
        parts = path_str.split("+")
        if len(parts) > 1:
            return int(parts[1])
        raise ValueError(f"Unable to parse DFDC label from path: {path}")

    for word in real_words:
        if word in path_str:
            return 0
    for word in fake_words:
        if word in path_str:
            return 1

    raise ValueError(f"Unable to determine label from path: {path}")


def process_videos(videos, imgs, labels):
    for video in tqdm.tqdm(videos, desc="  collect", leave=False):
        for frame in video:
            label = determine_labels(frame)
            frame = Path(str(frame).split("+")[0]) if "dfdc" in str(frame).lower() else Path(frame)
            imgs.append(frame)
            labels.append(label)


def split_videos(videos, test_size=0.2, random_state=42):
    """
    Stratified split at the video level.
    Splits real/fake videos separately so class balance is preserved.
    """
    real_videos, fake_videos = [], []

    for video in videos:
        if not video:
            continue
        label = determine_labels(video[0])
        if label == 0:
            real_videos.append(video)
        else:
            fake_videos.append(video)

    def _split(class_videos):
        if len(class_videos) == 0:
            return [], []
        if len(class_videos) == 1:
            return class_videos, []
        return train_test_split(
            class_videos,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    real_train, real_test = _split(real_videos)
    fake_train, fake_test = _split(fake_videos)

    train = real_train + fake_train
    test = real_test + fake_test

    random.shuffle(train)
    random.shuffle(test)

    print(
        f"  split → train: {len(real_train)} real, {len(fake_train)} fake | "
        f"test: {len(real_test)} real, {len(fake_test)} fake"
    )

    return train, test


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MyDataset(Dataset):
    def __init__(self, img_paths, img_labels, aug_prob=0.5, test=False):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.aug_prob = aug_prob
        self.test = test

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        frames_path = self.img_paths[index]
        label = self.img_labels[index]

        aug_apply = (not self.test) and (random.random() < self.aug_prob)

        if aug_apply:
            do_flip = random.random() < 0.5
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            do_blur = random.random() < 0.3
            sigma = random.uniform(0.1, 1.0)

        img = Image.open(frames_path).convert("RGB")
        img = TF.resize(img, (224, 224))

        if aug_apply:
            if do_flip:
                img = TF.hflip(img)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            if do_blur:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        label = torch.tensor(label, dtype=torch.long)
        return img, label


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    DINOv2 ViT-S backbone (last 2 blocks + norm unfrozen)
    + lightweight MLP classification head.

    Input : [B, 3, 224, 224]
    Output: [B, 2] logits
    """

    def __init__(
        self,
        vit_name="dinov2_vits14",
        feature_dim=384,
        num_classes=2,
        mlp_hidden=512,
        dropout=0.3,
    ):
        super().__init__()

        self.vit = torch.hub.load("facebookresearch/dinov2", vit_name)

        # Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False

        # Unfreeze last 2 transformer blocks
        if hasattr(self.vit, "blocks"):
            for block in self.vit.blocks[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Unfreeze final LayerNorm
        if hasattr(self.vit, "norm"):
            for p in self.vit.norm.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit.forward_features(x)

        cls_token = out["x_norm_clstoken"]
        patch_tokens = out["x_norm_patchtokens"]
        patch_mean = patch_tokens.mean(dim=1)

        feats = torch.cat([cls_token, patch_mean], dim=-1)
        logits = self.head(feats)
        return logits

    def vit_params(self):
        return [p for p in self.vit.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())


# ─────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────
# Train / eval helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, labels in tqdm.tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        # Step LR every optimizer update (correct for step-based schedule)
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for imgs, labels in tqdm.tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(imgs)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += loss.item()
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    if len(set(all_labels)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    seed_everything(42)

    # Paths
    base = Path("/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo")
    ff_json_path = base / "rearrange/dataset_json/FaceForensics++.json"
    celeb_json_path = base / "rearrange/dataset_json/Celeb-DF-v2.json"
    dfdc_json_path = base / "rearrange/dataset_json/DFDC.json"
    weight_save_path = base / "vit_weights"
    weight_save_path.mkdir(parents=True, exist_ok=True)
    best_ckpt = weight_save_path / "best_vit_checkpoint.pth"

    # Build image lists
    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []
    test_imgs, test_labels = [], []

    # FF++: keep provided test set untouched, split training videos into train/val
    ff_train_videos, ff_test_videos = load_json(ff_json_path)
    ff_train_pool, ff_val_videos = split_videos(ff_train_videos, test_size=0.15)
    process_videos(ff_train_pool, train_imgs, train_labels)
    process_videos(ff_val_videos, val_imgs, val_labels)
    process_videos(ff_test_videos, test_imgs, test_labels)
    print("Finished FF++")

    # # Celeb-DF-v2: split into train/val/test at video level
    # _, all_celeb = load_json(celeb_json_path)
    # celeb_train_pool, celeb_test_videos = split_videos(all_celeb, test_size=0.20)
    # celeb_train_videos, celeb_val_videos = split_videos(celeb_train_pool, test_size=0.15)
    # process_videos(celeb_train_videos, train_imgs, train_labels)
    # process_videos(celeb_val_videos, val_imgs, val_labels)
    # process_videos(celeb_test_videos, test_imgs, test_labels)
    # print("Finished Celeb-DF-v2")

    # # DFDC: split into train/val/test at video level
    # _, all_dfdc = load_json(dfdc_json_path)
    # dfdc_train_pool, dfdc_test_videos = split_videos(all_dfdc, test_size=0.20)
    # dfdc_train_videos, dfdc_val_videos = split_videos(dfdc_train_pool, test_size=0.15)
    # process_videos(dfdc_train_videos, train_imgs, train_labels)
    # process_videos(dfdc_val_videos, val_imgs, val_labels)
    # process_videos(dfdc_test_videos, test_imgs, test_labels)
    # print("Finished DFDC")

    print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)} | Test images: {len(test_imgs)}")

    # Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 32
    LR_VIT = 1e-5
    LR_HEAD = 1e-4
    WD = 1e-2
    NUM_WORKERS = 4
    PATIENCE = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets & loaders
    train_dataset = MyDataset(train_imgs, train_labels, aug_prob=0.5, test=False)
    val_dataset = MyDataset(val_imgs, val_labels, aug_prob=0.0, test=True)
    test_dataset = MyDataset(test_imgs, test_labels, aug_prob=0.0, test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = DeepfakeDetector(
        vit_name="dinov2_vits14",
        feature_dim=384,
        num_classes=2,
        mlp_hidden=512,
        dropout=0.3,
    ).to(device)

    # Optimizer: separate learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.vit_params(), "lr": LR_VIT, "weight_decay": WD},
            {"params": model.head_params(), "lr": LR_HEAD, "weight_decay": WD},
        ]
    )

    # Step-based cosine warmup + decay
    total_steps = EPOCHS * max(len(train_loader), 1)
    warmup_steps = 2 * max(len(train_loader), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=1e-4)

    best_auc = float("-inf")
    best_state_dict = None

    print("\n" + "─" * 60)
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        global_step += len(train_loader)

        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}"
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state_dict,
                    "best_auc": best_auc,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_ckpt,
            )
            print(f"  ✓ New best AUC={best_auc:.4f} — checkpoint saved to {best_ckpt}")

        if early_stop(val_auc if not np.isnan(val_auc) else -1.0):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Best checkpoint saved at: {best_ckpt}")

    # Load best checkpoint and evaluate on the untouched test set
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"Final test_loss={test_loss:.4f}  test_auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
