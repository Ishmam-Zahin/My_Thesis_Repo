import argparse
import importlib.util
import json
import logging
import random
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from torchvision import transforms

from helpers.dataset_loader import get_dataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def load_model_class(model_path: str, class_name: str):
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    spec = importlib.util.spec_from_file_location("dynamic_model_module", str(model_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {model_path}")
    logging.info(f"✅ Dynamically loaded model class '{class_name}' from {model_path}")
    return getattr(module, class_name)


def setup_run_logging(log_dir: str, run_name: str):
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                  logging.StreamHandler()],
        force=True
    )
    logging.info(f"=== Training Run Started: {run_name} ===")
    return run_dir


def save_hyperparams(config: dict, run_dir: Path):
    hyperparams_path = run_dir / "hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Hyperparameters saved to {hyperparams_path}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description="Graph-Only Deepfake Detection Training - Config Driven")
    parser.add_argument("--config", type=str, default="configs/train3.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    model_path = model_config.get('model_path')
    model_class_name = model_config.get('model_class')
    if not model_path or not model_class_name:
        raise ValueError("model_path and model_class must be defined in train3.yaml")

    model_name = Path(model_path).stem
    logging.info(f"Using model: {model_name}")

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    log_base = Path(config['paths']['log_dir']) / model_name
    log_base.mkdir(parents=True, exist_ok=True)
    run_dir = setup_run_logging(str(log_base), run_name)
    save_hyperparams(config, run_dir)

    json_root = config['data']['json_root']
    dataset_name = config['data']['dataset_name']
    transform = get_transforms()
    train_dataset, test_dataset = get_dataset(json_root, dataset_name, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = None
    if test_dataset is not None and len(test_dataset) > 0:
        val_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    criterion = nn.CrossEntropyLoss().to(device)
    logging.info("✅ Using CrossEntropyLoss for graph-only training")

    ModelClass = load_model_class(model_config['model_path'], model_config['model_class'])
    model = ModelClass(
        vit_name=model_config['vit_name'],
        feature_dim=384,
        dropout=model_config.get('dropout', 0.2),
        num_of_frames=config['training']['num_frames'],
        num_gcn_layers=model_config.get('num_gcn_layers', 3),
        num_clusters=model_config.get('num_clusters', 512),
        num_transformer_blocks=model_config.get('num_transformer_blocks', 1),
        num_heads=model_config.get('num_heads', 8),
        mlp_dim=model_config.get('mlp_dim', 512),
        tau_s=model_config.get('tau_s', 0.6),
        tau_t=model_config.get('tau_t', 0.6),
        graph_eps=model_config.get('graph_eps', 1e-4),
        local_window=model_config.get('local_window', 3),
        spectral_hidden_dim=model_config.get('spectral_hidden_dim', 64),
        vit_weight_path=model_config.get('vit_weight')
    ).to(device)

    if torch.cuda.is_available():
        summary(model, input_data=(torch.randn(1, config['training']['num_frames'], 3, 224, 224).to(device)), device="cuda")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(
        trainable_params,
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    checkpoint_base = Path(config['paths']['checkpoint_dir']) / model_name
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_base / "best_model.pth"
    last_ckpt_path = checkpoint_base / "last_checkpoint.pth"

    start_epoch = 0
    best_val_auc = 0.0
    epochs_no_improve = 0

    if config.get('resume', False) and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_auc = ckpt.get('best_val_auc', 0.0)
        logging.info(f"Resumed from epoch {start_epoch}")

    logging.info("=== Starting Training ===")

    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits, _, _ = model(videos)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            train_preds.extend(probs)
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, (np.array(train_preds) > 0.5).astype(int))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    videos = videos.to(device)
                    labels = labels.to(device)
                    logits, _, _ = model(videos)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))

            val_labels_np = np.array(val_labels)
            val_preds_np = (np.array(val_preds) > 0.5).astype(int)
            cm = confusion_matrix(val_labels_np, val_preds_np)
            tn, fp, fn, tp = cm.ravel()
            precision = precision_score(val_labels_np, val_preds_np, zero_division=0)
            recall = recall_score(val_labels_np, val_preds_np, zero_division=0)

            logging.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
                f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}"
            )

            scheduler.step(val_auc)

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
            }
            torch.save(ckpt, last_ckpt_path)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                ckpt['best_val_auc'] = best_val_auc
                torch.save(ckpt, best_model_path)
                logging.info(f"✅ New best model saved at epoch {epoch+1} with Val AUC={best_val_auc:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s)")
        else:
            logging.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}"
            )

    logging.info("=== Training Finished ===")


if __name__ == "__main__":
    main()