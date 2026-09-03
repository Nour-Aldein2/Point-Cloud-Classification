"""
To run this file, use something like
```
python feature_visualisation.py --ckpt checkpoints/best_model.pt --split test --method tsne
python -m DGCNN.results.feature_visualisation --ckpt DGCNN/checkpoints/best_model_2048_20.pt --split test --method tsne
```

For UMAP, install `umap-learn` and use `--method umap`.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

## -- Fix [Register DGCNN.config as "config" before importing networks.py because networks.py uses "from config import Config".]
from .. import config as config_module
sys.modules["config"] = config_module
# -- End Fix

from ..config import Config
from ..networks import DGCNN
from ..data_utils import PointCloudData

PALETTES = {
    "house_scape": ["#5d9781", "#ab3d66", "#73b0c9", "#7f6f8c", "#cad8c9",
                    "#2b6e72", "#75bcc1", "#d6a65c", "#b86b4b", "#566b8f"],
}


def load_trained_model(checkpoint_path: str | Path, cfg: Config):
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
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.classes


def extract_embeddings(model, loader, device):
    embeddings = []
    labels = []

    # Input to the final classifier is the learned penultimate embedding.
    def save_embedding(module, inputs):
        embeddings.append(inputs[0].detach().cpu())

    hook = model.classifier[-1].register_forward_pre_hook(save_embedding)

    with torch.inference_mode():
        model.eval()
        for batch in tqdm(loader, desc="Extracting embeddings", ncols=80, leave=False, colour="#5d9781"):
            batch = batch.to(cfg.train.device)

            points = batch.pos
            batch_idx = batch.batch
            y_true = batch.y.view(-1)

            model(points, batch_idx)
            labels.append(y_true.detach().cpu())

    hook.remove()

    if not embeddings:
        raise ValueError("No samples found for feature visualisation.")

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    return embeddings, labels


def reduce_embeddings(embeddings, method: str, seed: int):
    if len(embeddings) < 2:
        raise ValueError("At least two samples are required for feature visualisation.")

    if method == "tsne":
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2,
                       perplexity=perplexity,
                       init="pca",
                       learning_rate="auto",
                       random_state=seed)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires the `umap-learn` package. Install it with `pip install umap-learn`."
            ) from exc

        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method!r}")

    return reducer.fit_transform(embeddings)


def make_feature_plot(projection,
                      labels,
                      classes: dict,
                      method: str,
                      colours: tuple[str, ...] = tuple(PALETTES["house_scape"]),
                      save_path: str | Path = "./Figures",
                      filename: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 8))

    class_items = sorted(classes.items(), key=lambda item: item[1])

    if len(class_items) > len(colours):
        raise ValueError(f"Not enough colours for {len(class_items)} classes.")

    for i, (class_name, class_id) in enumerate(class_items):
        mask = labels == class_id

        ax.scatter(projection[mask, 0],
                   projection[mask, 1],
                   s=64,
                   alpha=0.60,
                   color=colours[i],
                   label=class_name.replace("_", " ").title())

    method_name = method.upper() if method == "umap" else "t-SNE"

    # Remove spines, axes, ticks, and tick labels
    ax.set_axis_off()

    ax.set_title(f"Learned Feature Space ({method_name}) -- 2048 Nodes", fontsize=14, pad=10)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.tight_layout()

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"feature_space_{method}_{timestamp}.png"
    filepath = save_dir / filename

    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved feature visualisation to: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise learned PointNet baseline embeddings.")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to where the checkpoint was saved.")
    parser.add_argument("--split", default="test",
                        help="The split you wish to visualise.")
    parser.add_argument("--method", choices=("tsne", "umap"), default="tsne",
                        help="Dimensionality reduction method.")
    args = parser.parse_args()

    checkpoint_path = args.ckpt
    split_name = args.split
    method = args.method

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_trained_model(checkpoint_path, cfg).to(device)
    loader, classes = prepare_data(cfg, split_name)

    embeddings, labels = extract_embeddings(model, loader, device)
    projection = reduce_embeddings(embeddings, method, cfg.data.seed)

    make_feature_plot(projection,
                      labels,
                      classes,
                      method,
                      save_path="./Figures",
                      filename=f"feature_space_{method}_2048.png")
