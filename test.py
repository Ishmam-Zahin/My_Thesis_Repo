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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from helpers.dataset_loader import get_dataset
from helpers.create_spatial_edges import get_spatial_edges


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
    log_file = run_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                  logging.StreamHandler()],
        force=True
    )
    logging.info(f"=== Test Run Started: {run_name} ===")
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
    parser = argparse.ArgumentParser(description="Deepfake Detection Testing - Config Driven")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--branch-ablation",
        type=str,
        choices=["none", "cls", "graph"],
        default="none",
        help="Run the model with one branch zeroed out to test contribution",
    )
    args = parser.parse_args()

    # Load test config
    with open(args.config, 'r') as f:
        test_config = yaml.safe_load(f)

    # Load train config (contains model params + pool)
    train_config_path = test_config['model'].get('train_config_path')
    if train_config_path and Path(train_config_path).exists():
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        logging.info(f"✅ Loaded shared config from train.yaml: {train_config_path}")
    else:
        train_config = {}

    # Merge: train_config has priority for model hyperparameters
    model_config = {**train_config.get('model', {}), **test_config.get('model', {})}

    # ==================== Logging & Setup ====================
    model_name = Path(model_config['model_path']).stem
    logging.info(f"Using model: {model_name}")

    seed = test_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(test_config.get('training', {}).get('device', 'cuda'))
    logging.info(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"test_run_{timestamp}"
    log_base = Path(test_config['paths']['log_dir']) / model_name
    log_base.mkdir(parents=True, exist_ok=True)
    run_dir = setup_run_logging(str(log_base), run_name)
    save_hyperparams(test_config, run_dir)

    # ==================== Transforms & Graph Edges ====================
    transform = get_transforms()
    training_cfg = test_config['training']
    num_frames = training_cfg['num_frames']
    knn_k = training_cfg.get('knn_k', 8)

    video_spatial_src_edges, video_spatial_dst_edges = get_spatial_edges(
        T=num_frames, N=256, K=knn_k
    )
    video_spatial_src_edges = video_spatial_src_edges.to(device)
    video_spatial_dst_edges = video_spatial_dst_edges.to(device)

    # ==================== Model Creation (using train.yaml params) ====================
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
        num_of_frames=num_frames,
        num_of_nodes_per_frame=256,
        num_of_temporal_edge_per_node=model_config.get('num_of_temporal_edge_per_node', 4),
        video_spatial_src_edges=video_spatial_src_edges,
        video_spatial_dst_edges=video_spatial_dst_edges,
        vit_weight_path=model_config.get('vit_weight')
    ).to(device)

    branch_ablation = None if args.branch_ablation == "none" else args.branch_ablation

    # ==================== Load Checkpoint ====================
    checkpoint_path = model_config.get('checkpoint_path')
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        logging.info(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        logging.warning("⚠️ No checkpoint_path provided!")

    # ==================== Pooling weights from train.yaml ====================
    pool_cfg = train_config.get('pool', {})
    lamda_min = pool_cfg.get('lamda_min', 0.05)
    lamda_ortho = pool_cfg.get('lamda_ortho', 0.005)

    # ==================== Testing Loop ====================
    json_root = test_config['data']['json_root']
    dataset_names = test_config['data'].get('dataset_names', [test_config['data'].get('dataset_name')])
    batch_size = training_cfg['batch_size']

    results = {}
    logging.info("=== Starting Testing ===")

    for dataset_name in dataset_names:
        logging.info(f"\n🔬 Testing on dataset: {dataset_name}")
        _, test_dataset = get_dataset(json_root, dataset_name, transform)

        if not test_dataset or len(test_dataset) == 0:
            logging.warning(f"⚠️ No test data for {dataset_name}. Skipping.")
            continue

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

        model.eval()
        test_loss = 0.0
        test_preds, test_labels = [], []

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc=f"Testing {dataset_name}"):
                videos = videos.to(device)
                labels = labels.to(device)

                try:
                    logits, mincut_loss, ortho_loss = model(videos, branch_ablation=branch_ablation)
                except TypeError:
                    logits, mincut_loss, ortho_loss = model(videos)

                loss = nn.CrossEntropyLoss()(logits, labels) + \
                       (lamda_min * mincut_loss) + (lamda_ortho * ortho_loss)
                test_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_preds.extend(probs)
                test_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_auc = roc_auc_score(test_labels, test_preds)
        binary_preds = (np.array(test_preds) > 0.5).astype(int)

        test_acc = accuracy_score(test_labels, binary_preds)
        test_precision = precision_score(test_labels, binary_preds, zero_division=0)
        test_recall = recall_score(test_labels, binary_preds, zero_division=0)
        cm = confusion_matrix(test_labels, binary_preds)
        tn, fp, fn, tp = cm.ravel()

        logging.info(
            f"✅ {dataset_name} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test AUC: {test_auc:.4f} | "
            f"Precision: {test_precision:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"Samples: {len(test_dataset)}"
        )
        logging.info(f"Confusion Matrix — TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

        results[dataset_name] = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "num_samples": len(test_dataset),
            "branch_ablation": branch_ablation or "none",
        }

    results_path = run_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"✅ All test results saved to {results_path}")
    logging.info("=== Testing Finished ===")

    if results:
        logging.info(f"Overall best AUC: {max(r['test_auc'] for r in results.values()):.4f}")


if __name__ == "__main__":
    main()