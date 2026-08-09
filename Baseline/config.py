from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DataConfig:
    path: str | Path = Path("/Users/nour/Downloads/ModelNet10/")

    num_points: int = 2048
    noise_sigma: float = 0.02
    seed: int = 11


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)


# TODO: How to augment 3D images?
# TODO: Consider merging train and test and resampling such that each class is 80:5:15
## TODO: Rotation should be done along different axes not only z!
## TODO: Consider reflection!
