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
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from torchvision import transforms

# ====================== Your existing modules ======================
from helpers.dataset_loader import get_dataset
from helpers.create_spatial_edges import get_spatial_edges


# ====================== Dynamic Model Loader ======================
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


# ====================== Logging Setup ======================
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


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training - Config Driven")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to train.yaml configuration file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ==================== Model Name & Logging ====================
    model_config = config.get('model', {})
    model_path = model_config.get('model_path')
    model_class_name = model_config.get('model_class')
    if not model_path or not model_class_name:
        raise ValueError("model_path and model_class must be defined in train.yaml")

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

    # ==================== Dataset ====================
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

    # ==================== Spatial Edges ====================
    video_spatial_src_edges, video_spatial_dst_edges = get_spatial_edges(
        T=config['training']['num_frames'],
        N=256,
        K=config['training']['knn_k']
    )
    video_spatial_src_edges = video_spatial_src_edges.to(device)
    video_spatial_dst_edges = video_spatial_dst_edges.to(device)

    # ==================== Model ====================
    ModelClass = load_model_class(model_config['model_path'], model_config['model_class'])

    model = ModelClass(
        vit_name=model_config['vit_name'],
        feature_dim=384,
        num_gcn_layers=model_config.get('num_gcn_layers', 2),
        num_clusters=model_config.get('num_clusters', 512),
        num_transformer_blocks=model_config.get('num_transformer_blocks', 1),
        num_heads=model_config.get('num_heads', 8),
        mlp_dim=model_config.get('mlp_dim', 512),
        dropout=model_config.get('dropout', 0.2),
        num_of_frames=config['training']['num_frames'],
        num_of_nodes_per_frame=256,
        num_of_temporal_edge_per_node=model_config.get('num_of_temporal_edge_per_node', 4),
        video_spatial_src_edges=video_spatial_src_edges,
        video_spatial_dst_edges=video_spatial_dst_edges,
        vit_weight_path=model_config.get('vit_weight')
    ).to(device)

    if torch.cuda.is_available():
        summary(model, input_data=(torch.randn(1, config['training']['num_frames'], 3, 224, 224).to(device)), device="cuda")

    # ==================== Optimizer with separate LR for ViT ====================
    vit_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vit" in name.lower():
            vit_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': vit_params, 'lr': config['training'].get('vit_lr', 1e-5)},
        {'params': other_params, 'lr': config['training']['lr']}
    ], weight_decay=config['training'].get('weight_decay', 1e-5))

    # ==================== Loss & Scheduler ====================
    real_weight = config['data'].get('real_weight', 5.0)
    fake_weight = config['data'].get('fake_weight', 1.0)
    class_weights = torch.tensor([real_weight, fake_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ==================== Checkpointing ====================
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

    # ==================== Training Loop ====================
    lamda_min = config['pool'].get('lamda_min', 0.1)
    lamda_ortho = config['pool'].get('lamda_ortho', 0.1)
    logging.info("=== Starting Training ===")

    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits, mincut_loss, ortho_loss = model(videos)

            loss = criterion(logits, labels)
            loss += (lamda_min * mincut_loss) + (lamda_ortho * ortho_loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            train_preds.extend(probs)
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, (np.array(train_preds) > 0.5).astype(int))

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    videos = videos.to(device)
                    labels = labels.to(device)
                    logits, mincut_loss, ortho_loss = model(videos)
                    loss = criterion(logits, labels) + (lamda_min * mincut_loss) + (lamda_ortho * ortho_loss)
                    val_loss += loss.item()
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_preds.extend(probs)
                    val_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        else:
            val_loss = train_loss
            val_auc = train_auc
            val_acc = train_acc

        scheduler.step(val_auc)

        # Logging
        logging.info(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
        }, last_ckpt_path)

        # Save best model
        if val_auc > best_val_auc + config['early_stopping']['delta']:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
            }, best_model_path)
            logging.info(f"✅ New best model saved! Val AUC = {val_auc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['early_stopping']['patience']:
            logging.info("Early stopping triggered!")
            break

    logging.info(f"=== Training Finished === Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()