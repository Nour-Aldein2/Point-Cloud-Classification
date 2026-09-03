"""
To run this file, use something like
```
python evaluate.py --ckpt checkpoints/best_model.pt --split test 2>&1 | tee evaluate_baseline.log
```
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm

from config import Config
from model import PointNetBaseline
from data_utils import PointCloudData


def load_trained_model(checkpoint_path: str | Path, cfg: Config):
    hist_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model_state_dict = hist_dict["model_state_dict"]

    model = PointNetBaseline(cfg)
    model.load_state_dict(model_state_dict)

    return model


def prepare_data(cfg: Config, split_name: str):
    dataset = PointCloudData(root_dir=cfg.data.path,
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
                          colours: tuple[str, ...] = ("#ffffff", "#e9f0f0", "#d4e2e2", "#bfd3d4", "#aac5c6", "#95b6b8",
                                                      "#7fa8aa", "#6a999c", "#558b8e", "#407c80", "#2b6e72"),
                          style: Literal["heatmap", "scatter"] = "heatmap",
                          colour_map: Literal["existing", "red"] = "existing",
                          save_path: str | Path = "../../Figures",
                          filename: str | None = None):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize per row (recall per class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix.astype(float), row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    # Colour map
    if colour_map == "existing":
        cmap = LinearSegmentedColormap.from_list("existing_cmap", colours, N=256)
    elif colour_map == "red":
        full_red = LinearSegmentedColormap.from_list("full_red", ["#F8D4D0", "#B2182B"], N=256)
        red_colours = full_red(np.linspace(0.0, 0.60, 256))
        cmap = LinearSegmentedColormap.from_list("reduced_red", red_colours, N=256)
    else:
        raise ValueError("colour_map must be 'existing' or 'red'.")

    norm = Normalize(vmin=0.0, vmax=1.0)
    target_names = [name.replace("_", " ").title() for name in target_names]

    fig, ax = plt.subplots(figsize=(10, 8))

    if style == "heatmap":
        im = ax.imshow(matrix_norm, interpolation="nearest", cmap=cmap, norm=norm)

        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.ax.set_ylabel("Recall", rotation=-90, va="bottom", fontsize=11)

        thresh = matrix_norm.max() / 2.0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                norm_val = matrix_norm[i, j]
                raw_val = matrix[i, j]
                text_colour = "white" if norm_val > thresh else "black"

                ax.text(j, i - 0.12, f"{norm_val:.2f}", ha="center", va="center", color=text_colour, fontsize=9)
                ax.text(j, i + 0.18, f"({raw_val})", ha="center", va="center", color=text_colour, fontsize=8, alpha=0.6)

        x_rotation = 45
        x_alignment = "right"

    elif style == "scatter":
        n_classes = len(labels)
        idx = np.arange(n_classes)
        xx, yy = np.meshgrid(idx, idx)

        x = xx.ravel()
        y = yy.ravel()
        recall_values = matrix_norm.ravel()

        # Recall controls both marker size and colour
        marker_sizes = 8 + 900 * recall_values

        scatter = ax.scatter(x, y, s=marker_sizes, c=recall_values, cmap=cmap, norm=norm, edgecolors="0.7", linewidths=0.65, zorder=3)

        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(True, linewidth=0.8, alpha=0.28)

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.022, pad=0.045, aspect=40)
        cbar.set_label("Recall", rotation=270, labelpad=18)
        cbar.set_ticks(np.linspace(0, 1, 6))

        ax.set_xlim(-0.5, n_classes - 0.5)
        ax.set_ylim(n_classes - 0.5, -0.5)
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis="both", length=0)

        x_rotation = 90
        x_alignment = "right"

    else:
        raise ValueError("style must be 'heatmap' or 'scatter'.")

    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=target_names,
           yticklabels=target_names, xlabel="Predicted Label", ylabel="True Label")
    plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha=x_alignment, rotation_mode="anchor")

    if filename is not None:
        try:
            node_count = Path(filename).stem.split("_")[2]
            title = f"Confusion Matrix (Exp. {node_count} Nodes)"
        except IndexError:
            title = "Confusion Matrix"
    else:
        title = "Confusion Matrix"

    ax.set_title(title, fontsize=14, pad=15)
    fig.tight_layout()

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"confusion_matrix_{style}_{colour_map}_{timestamp}.png"

    filepath = save_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved confusion matrix to: {filepath}")
    return matrix


if __name__ == "__main__":
    print("🚨 Important: Remember to use `2>&1 | tee evaluate.log` to store the logs")

    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet baseline checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to where the checkpoint was saved.")
    parser.add_argument("--split", required=True, help="The split you wish to evaluate the model on.")
    args = parser.parse_args()
    checkpoint_path = args.ckpt
    split_name = args.split

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = load_trained_model(checkpoint_path, cfg)
    loader, classes = prepare_data(cfg, split_name)

    model = model.to(device)

    results = {"y_true": [], "y_pred": []}
    with torch.inference_mode():
        model.eval()
        for batch in tqdm(loader, desc=f"Evaluating {split_name} split", ncols=80, leave=False, colour="#5d9781"):
            points = batch["point_cloud"].to(device)
            y_true = batch["category"].to(device)

            logits = model(points)
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
                          style="scatter",
                          colour_map="red",
                          save_path="../Figures",
                          filename=f"confusion_matrix_baseline.png")