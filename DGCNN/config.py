##
## TODO: since the baseline was trained with Adam optimizer, and the paper uses SGD, DGCCNN should be trained with both (as a uniques experiment).
## TODO: Check how to measure the time for a forward pass
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Data:
    num_points: int = 2048   # Number of points sampled in the shape
    k: int = 40   # Number of nearest neighbours to a point


@dataclass
class Architecture:
    pass


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


