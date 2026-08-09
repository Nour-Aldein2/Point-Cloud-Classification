import torch
from torch import nn
import torch.nn.functional as F

from config import Config


class PointNetBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        cfg = cfg.model
        self.cfg = cfg

        # Shared point-wise MLP
        self.conv1 = nn.Conv1d(3, cfg.hidden_dim1, kernel_size=cfg.kernel_size)
        self.conv2 = nn.Conv1d(cfg.hidden_dim1, cfg.hidden_dim2, kernel_size=cfg.kernel_size)
        self.conv3 = nn.Conv1d(cfg.hidden_dim2, cfg.hidden_dim3, kernel_size=cfg.kernel_size)

        self.bn1 = nn.BatchNorm1d(cfg.hidden_dim1)
        self.bn2 = nn.BatchNorm1d(cfg.hidden_dim2)
        self.bn3 = nn.BatchNorm1d(cfg.hidden_dim3)

        # Classification head
        self.fc1 = nn.Linear(cfg.hidden_dim3, cfg.hidden_dim2)
        self.fc2 = nn.Linear(cfg.hidden_dim2, cfg.num_classes)

        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)   # (B, N, C) --> (B, C, N)   ......  C = 3

        # Shared MLP applied independently to every point
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))   # (B, hidden_dim3, N)

        # Global symmetric aggregation over all points
        x = torch.max(x, dim=2).values   # (B, hidden_dim3)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)   # (B, num_classes)

        x = self.fc2(x)
        return x
