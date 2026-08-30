import argparse
from pathlib import Path

import torch

from ..config import Config
from ..networks import DGCNN
from ..data_utils import PointCloudData


def load_trained_model(checkpoint_path: str | Path):
    hist_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model_state_dict = hist_dict["model_state_dict"]

    model = DGCNN()
    model.load_state_dict(model_state_dict)

    return model


