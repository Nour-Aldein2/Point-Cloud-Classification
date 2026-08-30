import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..config import Config
from ..networks import DGCNN
from ..data_utils import PointCloudData


def load_trained_model(checkpoint_path: str | Path):
    hist_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
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
    return loader


def compute_scores(y_true: Tensor, y_pred: Tensor, labels: list):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    report = classification_report(y_true, y_pred, labels=labels)
    print("="*50)
    print(f"Classification Report:\n{report}\n")
    print("=" * 50)
    return report


def make_confusion_matrix(y_true: Tensor,
                          y_pred: Tensor,
                          labels: list,
                          colours: list = ["#2b6e72", "#75bcc1"],
                          save_path: str | Path = "../../Figures"):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize per row (recall per class)
    matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

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
        xticklabels=labels,
        yticklabels=labels,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_{timestamp}.png"
    filepath = save_dir / filename

    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved confusion matrix to: {filepath}")
    return matrix




if __name__ == "__main__":
    print("🚨 Important: Remember to use `| 2&>1  tee evaluate.log to store the logs")   ## TODO: Is there a way to automate that?
    checkpoint_path = argparse.ArgumentParser("--ckpt", description="Path to where the checkpoint was saved.", argument_default="")
    split_name = argparse.ArgumentParser("--split", description="The split you wish to evaluate the model on.")

    cfg = Config()
    model = load_trained_model(checkpoint_path)
    loader = prepare_data(cfg, split_name)

    model = model.to(cfg.train.device)


