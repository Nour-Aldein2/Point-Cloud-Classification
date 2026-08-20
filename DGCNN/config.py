##
## TODO: since the baseline was trained with Adam optimizer, and the paper uses SGD, DGCCNN should be trained with both (as a uniques experiment).
## TODO: Check how to measure the time for a forward pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch


@dataclass
class Data:
    root_dir: str | Path = Path("/Users/nour/Downloads/ModelNet10_3_splits/")
    num_points: int = 1024   # Number of points sampled in the shape
    k: int = 20   # Number of nearest neighbours to a point
    num_classes: int = 10

    noise_sigma: float = 0.02
    seed: int = 11

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = torch.cuda.is_available()


@dataclass
class Architecture:
    edge_channels: Sequence[int] = (64, 64, 128, 256)
    embedding_dim: int = 1024
    classifier_dims: Sequence[int] = (512, 256)

    agg_fcn: "str" = "max"
    self_loop: bool = True

    leaky_relu_slope: float = 0.2
    dropout_rate: float = 0.5
    batch_norm_momentum: float = 0.1  # "BN momentum 0.9" in the paper corresponds to PyTorch momentum=0.1


@dataclass
class Train:
    epochs: int = 250
    es_patience: int = 50
    save_path: str | Path = Path("./checkpoints")

    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.SGD
    lr_initial: float = 0.01
    lr_final: float = 0.0001
    momentum: float = 0.9
    weigh_decay: float = 1e-4

    # label_smooth: float = 0.2


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    architecture: Architecture = field(default_factory=Architecture)
    train: Train = field(default_factory=Train)


