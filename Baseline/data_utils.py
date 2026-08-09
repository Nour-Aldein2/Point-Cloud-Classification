import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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
        noise = np.random.uniform(0, self.sigma, point_cloud.shape)

        return point_cloud + noise


class ToTensor:
    def __call__(self, point_cloud):
        assert len(point_cloud.shape) == 2, f"Incorrect point cloud shape: {point_cloud.shape}"

        return torch.from_numpy(point_cloud)


class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train", transform=None):
        self.root_dir = root_dir

    def __len__(self):
        pass

    def __preproc__(self, file):
        pass

    def __getitem__(self, item):
        return


def default_transforms():
    return transforms.Compose([
        Normalize(),
        RandomRotationZ(),
        RandomNoise(),
        ToTensor()
    ])