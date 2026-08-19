from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import EdgeConv, knn_graph, global_mean_pool, global_max_pool
# IMPORTANT: You could achieve the implementation using DynamicEdgeConv instead of using both EdgeConv, knn_graph

from config import Config


class MLP(nn.Module, Config):
    """
    MLP used by EdgeConv.

    PyG EdgeConv constructs:
        cat([x_i, x_j - x_i], dim=-1)

    Therefore, a node representation with F channels produces an
    edge representation with 2 * F channels.
    """
    def __init__(self, in_channels, out_channels, slope):
        nn.Module.__init__(self)
        Config.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels*2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, momentum=self.architecture.batch_norm_momentum),
            nn.LeakyReLU(slope)
        )

    def forward(self, x):
        return self.mlp(x)


class DGCNN(nn.Module, Config):
    def __init__(self):
        nn.Module.__init__(self)
        Config.__init__(self)
        cfg = self.architecture
        mlp_dims = cfg.edge_channels

        mlp1 = MLP(3, mlp_dims[0], cfg.leaky_relu_slope)
        mlp2 = MLP(mlp_dims[0], mlp_dims[1], cfg.leaky_relu_slope)
        mlp3 = MLP(mlp_dims[1], mlp_dims[2], cfg.leaky_relu_slope)
        mlp4 = MLP(mlp_dims[2], mlp_dims[3], cfg.leaky_relu_slope)

        self.edge_conv1 = EdgeConv(nn=mlp1, aggr=cfg.agg_fcn)
        self.edge_conv2 = EdgeConv(nn=mlp2, aggr=cfg.agg_fcn)
        self.edge_conv3 = EdgeConv(nn=mlp3, aggr=cfg.agg_fcn)
        self.edge_conv4 = EdgeConv(nn=mlp4, aggr=cfg.agg_fcn)

        self.point_embedding = nn.Sequential(
            nn.Linear(sum(mlp_dims), cfg.embedding_dim, bias=False),
            nn.BatchNorm1d(cfg.embedding_dim, momentum=cfg.batch_norm_momentum),
            nn.LeakyReLU(cfg.leaky_relu_slope)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2*cfg.embedding_dim, cfg.classifier_dims[0], bias=False),   # Max and mean pooling each produce emb_dims channels, so the classifier receives 2 * emb_dims.
            nn.LeakyReLU(cfg.leaky_relu_slope),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.classifier_dims[0], cfg.classifier_dims[1]),
            nn.LeakyReLU(cfg.leaky_relu_slope),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.classifier_dims[1], self.data.num_classes)
        )

    def forward(self, x: Tensor, batch: Optional[Tensor]):
        """
        Args:
            x:
                Point coordinates/features with shape
                [total_points_in_batch, in_channels].

            batch:
                Graph membership vector with shape
                [total_points_in_batch].

        Returns:
            Raw classification logits with shape
            [num_graphs, num_classes].
        """

        edge_index1 = knn_graph(x, self.data.k, batch=batch, loop=self.architecture.self_loop)
        h1 = self.edge_conv1(x, edge_index1)

        edge_index2 = knn_graph(h1, self.data.k, batch=batch, loop=self.architecture.self_loop)
        h2 = self.edge_conv2(h1, edge_index2)

        edge_index3 = knn_graph(h2, self.data.k, batch=batch, loop=self.architecture.self_loop)
        h3 = self.edge_conv3(h2, edge_index3)

        edge_index4 = knn_graph(h3, self.data.k, batch=batch, loop=self.architecture.self_loop)
        h4 = self.edge_conv4(h3, edge_index4)

        h = torch.cat([h1, h2, h3, h4], dim=1)   # [P, 64 + 64 + 128 + 256] = [P, 512]
        h = self.point_embedding(h)   # [P, 512] -> [P, emb_dims]

        # Each has shape [B, emb_dims]. graph-level representation
        h_max = global_max_pool(h, batch)
        h_mean = global_mean_pool(h, batch)

        h = torch.cat((h_max, h_mean), dim=1)   # [B, 2 * emb_dims]

        # Classifier (Raw logits)
        return self.classifier(h)
