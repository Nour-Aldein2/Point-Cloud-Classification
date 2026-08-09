from pathlib import Path
from typing import Callable
import random

import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm


class Normalize:
    def __call__(self, point_cloud):
        assert len(point_cloud.shape) == 2

        norm_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))

        return norm_point_cloud


class RandomRotationZ:
    def __call__(self, point_cloud):
        theta = np.random.uniform(0, 2*np.pi)

        transform = trimesh.transformations.rotation_matrix(theta, [0, 0, 1])

        return trimesh.transform_points(point_cloud, transform)


class RandomNoise:
    def __init__(self, sigma: float = 0.02):
        self.sigma = sigma

    def __call__(self, point_cloud):
        # noise = np.random.uniform(-self.sigma, self.sigma, point_cloud.shape)   # uniform jitter (symmetric)
        noise = np.random.uniform(0, self.sigma, point_cloud.shape)   # Gaussian jitter
        return point_cloud + noise


class ToTensor:
    def __call__(self, point_cloud):
        assert len(point_cloud.shape) == 2, f"Incorrect point cloud shape: {point_cloud.shape}"

        return torch.from_numpy(point_cloud).float()


class PointCloudData(Dataset):
    def __init__(self, root_dir: Path, num_points: int, split_name: str = "train", transform: Callable | None = None, sigma: float = 0.02, seed: int = 42):
        self.root_dir = root_dir
        folders = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
        self.classes = {f.stem: i for i, f in enumerate(folders)}
        self.num_points = num_points
        self.split_name = split_name
        self.sigma = sigma
        self.seed = seed

        self.transforms = transform if transform is not None else self.default_transforms()

        self.files = []
        for category_path in tqdm(folders, desc=f"Preparing dataset for {split_name} split"):
            data_path = category_path/split_name
            for obj_path in sorted(data_path.glob("*.off")):
                sample = {
                    "pcd_path": obj_path,
                    "category": category_path.name
                }
                self.files.append(sample)
        random.seed(seed)
        random.shuffle(self.files)

    def default_transforms(self):
        if self.split_name == "train":
            return transforms.Compose([
                Normalize(),
                RandomRotationZ(),
                RandomNoise(self.sigma),
                ToTensor()
            ])
        else:
            return transforms.Compose([
                Normalize(),
                ToTensor()
            ])

    def __len__(self):
        return len(self.files)

    def _sample_points(self, mesh, idx):
        if self.split_name == "train":
            # New surface sample each time the training
            # object is requested.
            seed = None
        else:
            # Deterministic evaluation point cloud.
            seed = self.seed + idx

        point_cloud, _ = trimesh.sample.sample_surface(
            mesh,
            count=self.num_points,
            seed=seed,
        )

        return point_cloud

    def __getitem__(self, idx):
        sample = self.files[idx]

        pcd_path = sample["pcd_path"]
        category = sample["category"]

        mesh = trimesh.load_mesh(pcd_path)

        point_cloud = self._sample_points(mesh, idx)
        point_cloud = self.transforms(point_cloud)

        label = torch.tensor(self.classes[category], dtype=torch.long)

        return {
            "point_cloud": point_cloud,
            "category": label,
        }
