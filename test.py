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
# ====================== Your existing modules ======================
from helpers.dataset_loader import get_dataset
from helpers.create_spatial_edges import get_spatial_edges
# ====================== Dynamic Model Loader ======================
def load_model_class(model_path: str, class_name: str):
    """Dynamically load a model class from a .py file path (same as train.py)."""
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
    """Create timestamped folder + log file + hyperparams.json (same as train.py)."""
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log" # keep same filename for consistency
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"=== Test Run Started: {run_name} ===")
    return run_dir
def save_hyperparams(config: dict, run_dir: Path):
    """Save full config as JSON for reproducibility (same as train.py)."""
    hyperparams_path = run_dir / "hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Hyperparameters saved to {hyperparams_path}")
def get_transforms():
    """Same transforms as train.py."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Testing - Config Driven")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to test.yaml configuration file")
    args = parser.parse_args()
    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # ==================== Determine Model Name for Organized Logging ====================
    model_config = config.get('model', {})
    model_path = model_config.get('model_path')
    model_class_name = model_config.get('model_class')
    vit_name = model_config.get('vit_name')
    checkpoint_path = model_config.get('checkpoint_path')
    if not model_path or not model_class_name or not vit_name:
        raise ValueError(
            "Configuration must include 'model_path', 'model_class', and 'vit_name' under the 'model' section.\n"
            "Also provide 'checkpoint_path' for testing.\n"
            "Example:\n"
            "model:\n"
            " model_path: '/path/to/zahin_model_video.py'\n"
            " model_class: 'MyModel'\n"
            " vit_name: 'dinov2_vits14'\n"
            " checkpoint_path: '/path/to/best_model.pth'"
        )
    model_name = Path(model_path).stem # e.g. "zahin_model_video"
    logging.info(f"Using model folder: {model_name} (from {model_path})")
    # ==================== Seed & Device ====================
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Using device: {device}")
    # ==================== Timestamped Run Folder (inside model-specific folder) ====================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"test_run_{timestamp}" # prefixed so you can distinguish from training runs
    log_base = Path(config['paths']['log_dir']) / model_name
    log_base.mkdir(parents=True, exist_ok=True)
    run_dir = setup_run_logging(str(log_base), run_name)
    save_hyperparams(config, run_dir)
    # ==================== Transforms & Graph (same as train.py) ====================
    transform = get_transforms()
    training_cfg = config['training']
    num_frames = training_cfg['num_frames']
    knn_k = training_cfg.get('knn_k', 4) # must match what you used during training
    #getting spatial edges for one video
    video_spatial_src_edges, video_spatial_dst_edges = get_spatial_edges(
        T=num_frames,
        N=256,
        K=knn_k
    )
    video_spatial_src_edges = video_spatial_src_edges.to(device)
    video_spatial_dst_edges = video_spatial_dst_edges.to(device)
    # ==================== Dynamic Model + Load Checkpoint ====================
    ModelClass = load_model_class(model_path, model_class_name)
    model = ModelClass(
        vit_name=vit_name,
        video_spatial_src_edges = video_spatial_src_edges,
        video_spatial_dst_edges = video_spatial_dst_edges
    ).to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        logging.info(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        logging.warning("⚠️ No checkpoint_path provided. Using randomly initialized model!")
    # ==================== Testing Loop over multiple datasets ====================
    json_root = config['data']['json_root']
    dataset_names = config['data']['dataset_names'] # list of dataset names
    batch_size = training_cfg['batch_size']
    real_weight = 5.0
    fake_weight = 1.0

    class_weights = torch.tensor([real_weight, fake_weight]).to(device)

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    results = {}
    logging.info("=== Starting Testing ===")
    for dataset_name in dataset_names:
        logging.info(f"\n🔬 Testing on dataset: {dataset_name}")
        # get_dataset returns (train_dataset, test_dataset)
        # For non-FF++ cross-dataset JSONs: train_dataset=None, test_dataset=full dataset
        # For FF++: test_dataset=the test split
        _, test_dataset = get_dataset(json_root, dataset_name, transform)
        if test_dataset is None or len(test_dataset) == 0:
            logging.warning(f"⚠️ No test data found for {dataset_name}. Skipping.")
            continue
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        # ----------------- Evaluation -----------------
        model.eval()
        test_loss = 0.0
        test_preds, test_labels = [], []

        lamda_min = 0.1
        lamda_ortho = 0.1

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc=f"Testing {dataset_name}"):
                videos = videos.to(device)
                labels = labels.to(device)
                logits, mincut_loss, ortho_loss = model(videos)
                loss = criterion(logits, labels)
                loss += (lamda_min * mincut_loss) + (lamda_ortho * ortho_loss)
                test_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                test_preds.extend(probs)
                test_labels.extend(labels.cpu().numpy())
        test_loss /= len(test_loader)
        test_auc = roc_auc_score(test_labels, test_preds)
        binary_preds = (np.array(test_preds) > 0.7).astype(int)
        test_acc = accuracy_score(test_labels, binary_preds)
        test_precision = precision_score(test_labels, binary_preds, zero_division=0)
        test_recall = recall_score(test_labels, binary_preds, zero_division=0)
        cm = confusion_matrix(test_labels, binary_preds)
        tn, fp, fn, tp = cm.ravel()
        # Log per-dataset results
        logging.info(
            f"✅ {dataset_name} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test AUC: {test_auc:.4f} | "
            f"Precision: {test_precision:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"Samples: {len(test_dataset)}"
        )
        logging.info(
            f"Confusion Matrix — TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}"
        )
        results[dataset_name] = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "num_samples": len(test_dataset)
        }
    # ==================== Save Results ====================
    results_path = run_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"✅ All test results saved to {results_path}")
    logging.info("=== Testing Finished ===")
    if results:
        logging.info(f"Overall best AUC across datasets: {max(r['test_auc'] for r in results.values()):.4f}")
if __name__ == "__main__":
    main()