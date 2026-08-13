##
## TODO: since the baseline was trained with Adam optimizer, and the paper uses SGD, DGCCNN should be trained with both (as a uniques experiment).
## TODO: Check how to measure the time for a forward pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch


@dataclass
class Data:
    num_points: int = 2048   # Number of points sampled in the shape
    k: int = 40   # Number of nearest neighbours to a point
    num_classes: int = 10


@dataclass
class Architecture:
    edge_channels: Sequence[int] = (64, 64, 128, 256)
    embedding_dim: int = 1024
    classifier_dims: Sequence[int] = (512, 256)

    agg_fcn: "str" = "max"
    self_loop: bool = True

    leaky_relu_slope: float = 0.2
    dropout_rate: float = 0.5


@dataclass
class Train:
    batch_size: int = 32
    optimizer = torch.optim.SGD
    lr_initial: float = 0.1
    lr_final: float = 0.001
    batch_norm_momentum: float = 0.9
    momentum: float = 0.9


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    architecture: Architecture = field(default_factory=Architecture)
    train: Train = field(default_factory=Train)


