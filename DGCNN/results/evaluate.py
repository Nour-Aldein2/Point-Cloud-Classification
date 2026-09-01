import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from ..config import Config
from ..networks import DGCNN
from ..data_utils import PointCloudData


def load_trained_model(checkpoint_path: str | Path):
    hist_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model_state_dict = hist_dict["model_state_dict"]

    model = DGCNN()
    model.load_state_dict(model_state_dict)

    return model


def prepare_data(cfg: Config, split_name: str):
    dataset = PointCloudData(root_dir=cfg.data.root_dir,
                             num_points=cfg.data.num_points,
                             split_name=split_name,
                             seed=cfg.data.seed)

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return loader, dataset.classes


def compute_scores(y_true: Tensor, y_pred: Tensor, labels: list, target_names: list):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    report = classification_report(y_true,
                                   y_pred,
                                   labels=labels,
                                   target_names=target_names,
                                   zero_division=0)

    print("="*50)
    print(f"Classification Report:\n{report}\n")
    print("=" * 50)
    return report


def make_confusion_matrix(y_true: Tensor,
                          y_pred: Tensor,
                          labels: list,
                          target_names: list,
                          colours: tuple[str, str] = ("#2b6e72", "#75bcc1"),
                          save_path: str | Path = "../../Figures"):  # -- End Fix
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize per row (recall per class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(
        matrix.astype('float'),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap from your two colours
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colours)

    im = ax.imshow(matrix_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.set_ylabel('Recall', rotation=-90, va="bottom", fontsize=11)

    # Ticks & labels
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        ## -- Fix [Display class names on the axes while retaining numeric IDs for the confusion-matrix calculation.]
        xticklabels=target_names,
        yticklabels=target_names,
        # -- End Fix
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with count (raw) and proportion
    thresh = matrix_norm.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            norm_val = matrix_norm[i, j]
            raw_val = matrix[i, j]
            text = f"{raw_val}\n({norm_val:.2f})"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if norm_val > thresh else "black",
                    fontsize=9)

    ax.set_title("Confusion Matrix", fontsize=14, pad=15)
    fig.tight_layout()

    # --- Save ---
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate filename with timestamp to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"confusion_matrix_{timestamp}.png"
    filepath = save_dir / filename

    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved confusion matrix to: {filepath}")
    return matrix


if __name__ == "__main__":
    print("🚨 Important: Remember to use `2>&1 | tee evaluate.log` to store the logs")

    parser = argparse.ArgumentParser(description="Evaluate a trained DGCNN checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to where the checkpoint was saved.")
    parser.add_argument("--split", required=True, help="The split you wish to evaluate the model on.")
    args = parser.parse_args()
    checkpoint_path = args.ckpt
    split_name = args.split

    cfg = Config()
    model = load_trained_model(checkpoint_path)
    loader, classes = prepare_data(cfg, split_name)

    model = model.to(cfg.train.device)

    results = {"y_true": [], "y_pred": []}
    with torch.inference_mode():
        model.eval()
        for batch in tqdm(loader, desc=f"Evaluating {split_name} split", ncols=80, leave=False, colour="#5d9781"):
            batch = batch.to(cfg.train.device)

            points = batch.pos
            batch_idx = batch.batch
            y_true = batch.y.view(-1)

            logits = model(points, batch_idx)
            y_pred = logits.argmax(dim=1)

            results["y_true"].append(y_true.detach().cpu())
            results["y_pred"].append(y_pred.detach().cpu())

    if not results["y_true"]:
        raise ValueError(f"No samples found in the {split_name!r} split.")

    y_true = torch.cat(results["y_true"], dim=0)
    y_pred = torch.cat(results["y_pred"], dim=0)

    class_items = sorted(classes.items(), key=lambda item: item[1])
    labels = [class_id for _, class_id in class_items]
    target_names = [class_name for class_name, _ in class_items]

    compute_scores(y_true, y_pred, labels, target_names)
    make_confusion_matrix(y_true,
                          y_pred,
                          labels,
                          target_names,
                          save_path="../../Figures")
