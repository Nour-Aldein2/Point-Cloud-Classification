from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DataConfig:
    path: str | Path = Path("/Users/nour/Downloads/ModelNet10/")

    num_points: int = 2048
    noise_sigma: float = 0.02
    seed: int = 11

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class Model:
    num_classes: int = 10

    hidden_dim1: int = 64
    hidden_dim2: int = 128
    hidden_dim3: int = 256

    kernel_size: int = 1

    dropout_rate: float = 0.3

class Training:
    epochs: int = 1000
    es_patience: int = 50

    learning_rate: float = 1e-3

    save_path: str | Path = Path("./checkpoints")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)


# TODO: How to augment 3D images?
# TODO: Consider merging train and test and resampling such that each class is 80:5:15
## TODO: Rotation should be done along different axes not only z!
## TODO: Consider reflection!
## TODO: Develop voxelisation
# TODO: Try using other GNN layers other than EdgeConv