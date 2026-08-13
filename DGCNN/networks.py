import torch
from torch import nn
from torch_geometric.nn import EdgeConv, knn_graph, global_mean_pool, global_max_pool

# IMPORTANT: You could achieve the implementation using DynamicEdgeConv instead of using both EdgeConv, knn_graph

from Classification_ModelNet10.DGCNN.config import Config


class SpatialTransform(nn.Module):
    def __init__(self):
        super().__init__()


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, slope):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(slope)
        )

    def forward(self, x):
        return self.mlp(x)


class DGCNN(nn.Module, Config):
    def __init__(self):
        super().__init__()
        cfg = self.architecture
        spatial_transform = SpatialTransform

        mlp1 = MLP(3, cfg.mlp_out_dim_1, cfg.leaky_relu_slope)
        mlp2 = MLP(cfg.mlp_out_dim_1, cfg.mlp_out_dim_2, cfg.leaky_relu_slope)
        mlp3 = MLP(cfg.mlp_out_dim_2, cfg.mlp_out_dim_3, cfg.leaky_relu_slope)
        mlp4 = MLP(cfg.mlp_out_dim_3, cfg.mlp_out_dim_4, cfg.leaky_relu_slope)

        self.edge_conv1 = EdgeConv(nn=mlp1, aggr="max")
        self.edge_conv2 = EdgeConv(nn=mlp2, aggr="max")
        self.edge_conv3 = EdgeConv(nn=mlp3, aggr="max")
        self.edge_conv4 = EdgeConv(nn=mlp4, aggr="max")

        self.fc = nn.Sequential(
            nn.Linear(cfg.mlp_out_dim_1 + cfg.mlp_out_dim_2 + cfg.mlp_out_dim_3 + cfg.mlp_out_dim_4, cfg.fc_out_dim),
            nn.BatchNorm1d(cfg.fc_out_dim),
            nn.LeakyReLU(cfg.leaky_relu_slope)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.fc_out_dim, cfg.classifier_hidden1),
            nn.LeakyReLU(cfg.leaky_relu_slope),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.classifier_hidden1, cfg.classifier_hidden2),
            nn.LeakyReLU(cfg.leaky_relu_slope),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.classifier_hidden2, self.data.num_classes)
        )

    def forward(self, x, edge_index, batch):
        edge_index1 = knn_graph(x, self.data.k, batch=batch)
        x1 = self.edge_conv1(x, edge_index1)

        edge_index2 = knn_graph(x1, self.data.k, batch=batch)
        x2 = self.edge_conv2(x, edge_index2)

        edge_index3 = knn_graph(x2, self.data.k, batch=batch)
        x3 = self.edge_conv3(x, edge_index3)

        edge_index4 = knn_graph(x3, self.data.k, batch=batch)
        x4 = self.edge_conv4(x, edge_index4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fc(x)

        # graph-level representation
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        x = torch.cat((x_max, x_mean), dim=1)

        # Classifier
        return self.classifier(x)
