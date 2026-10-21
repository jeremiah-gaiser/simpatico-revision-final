from typing import Optional

import torch
from torch.nn import Linear, Sequential, SiLU
from torch_geometric.nn import GATv2Conv, knn, radius
from torch_geometric.nn import Sequential as PyG_Sequential


class PositionalEdgeGenerator(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        # Use Sigmoid at the end to bound edge weights between 0 and 1
        self.sigmoid = torch.nn.Sigmoid()
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            self.relu,
            torch.nn.Linear(hidden_dim, 1),
            self.sigmoid, # CHANGED: from relu to sigmoid
        )

    def forward(self, pos, x_subset, y_subset, r, batch):
        device = pos.device
        connections = radius(pos[x_subset], pos[y_subset], r, batch[x_subset], batch[y_subset], max_num_neighbors=32)

        edge_index = torch.vstack((y_subset[connections[0]], x_subset[connections[1]])).to(device)

        node_distances = (
            torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            .unsqueeze(1)
            .to(device)
        )

        edge_weights = self.MLP(node_distances)
        return edge_index, edge_weights

# class PositionalEdgeGenerator(torch.nn.Module):
#     """
#     Generates knn-based graph of edges between positional nodes and applies learnable non-linear layer to produce non-negative edge weights based on distance

#     Args:
#         hidden_dim (int): dimension of hidden layer in edge MLP.

#     Attributes:
#         MLP (Sequential[Linear, relu, Linear, relu]): produces edge weight values from distances
#     """

#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.relu = torch.nn.ReLU()
#         self.MLP = Sequential(
#             Linear(1, hidden_dim),
#             self.relu,
#             Linear(hidden_dim, 1),
#             self.relu,
#         )

#     def forward(
#         self,
#         pos: torch.Tensor,
#         x_subset: torch.Tensor,
#         y_subset: torch.Tensor,
#         k: int,
#         batch: torch.Tensor,
#     ):
#         """
#         Args:
#             pos (torch.Tensor): Nx3 tensor of node positions
#             x_subset (torch.Tensor): index of potential neighbor nodes
#             y_subset: (torch.Tensor): index of nodes for which we want to find neighbors
#             k (int): closest neighbor count
#             batch (torch.Tensor): PyG graph batch index

#         Returns:
#             (torch.Tensor, torch.Tensor): edge index and distance-based edges of new knn-graph
#         """
#         device = pos.device
#         connections = knn(
#             pos[x_subset], pos[y_subset], k, batch[x_subset], batch[y_subset]
#         )
#         edge_index = torch.vstack(
#             (y_subset[connections[0]], x_subset[connections[1]])
#         ).to(device)
#         node_distances = (
#             torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
#             .unsqueeze(1)
#             .to(device)
#         )
#         edge_weights = self.MLP(node_distances)
#         return edge_index, edge_weights


class ResLayer(torch.nn.Module):
    """
    Refactored ResLayer with LayerNorm and Pre-Activation structure.
    """
    def __init__(
        self,
        dims: int,
        act: torch.nn.Module = torch.nn.SiLU,
        dr: float = 0.1, # Increased default dropout slightly
        heads: int = 4,
        edge_dim: Optional[int] = None,
        norm: Optional[torch.nn.Module] = torch.nn.LayerNorm, # CHANGED: Default to LayerNorm
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.act = act()
        input_dim = dims * heads

        # add_self_loops: GATv2Conv calls remove_self_loops() BEFORE add_self_loops(), so there
        # is exactly one self-loop per node either way -- it does NOT stack on top of the
        # self-pairs that PositionalEdgeGenerator's radius(x, x, ...) produces on the protein
        # graph (658 atom-atom, 116 vox-vox in a typical pocket). The only difference is the
        # self-loop's edge feature: fill_value='mean' when True, versus the explicit MLP(0)
        # ~= 0.57 when False. Measured on ts005_th005, that changes nothing -- perturbation
        # response 0.01807 vs 0.01809, identity R^2 0.1180 both ways. Default stays True; the
        # flag exists only to A/B it. Ligand graphs carry no explicit self-loops at all, so
        # setting it False there would remove self-attention rather than deduplicate it.
        self.layer = GATv2Conv(input_dim, dims, heads=heads, edge_dim=edge_dim,
                               add_self_loops=add_self_loops)
        self.dropout = torch.nn.Dropout(dr)

        # Norm is applied to the input dimension
        self.norm = norm(input_dim) if norm is not None else torch.nn.Identity()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:

        # 1. Pre-Norm: Normalize x before passing to layer (stabilizes gradients)
        x_norm = self.norm(x)

        # 2. Convolution
        h = self.layer(x_norm, edge_index, edge_attr)

        # 3. Activation
        h = self.act(h)

        # 4. Dropout
        h = self.dropout(h)

        # 5. Residual Connection
        return x + h

# class ResLayer(torch.nn.Module):
#     """
#     A residual layer that performs graph attention convolution (GATv2) and adds the result to the input.

#     Args:
#         dims (int): The dimensionality of the input features (per head).
#         act (torch.nn.Module, optional): The activation function to use. Defaults to SiLU.
#         dr (float, optional): The dropout rate. Defaults to 0.25.
#         heads (int, optional): The number of attention heads in the GATv2 layer. Defaults to 4.
#         edge_dim (int, optional): Number of dimensions in edge attributes.
#         norm (torch.nn.Module, optional): The normalization layer applied before activation.

#     Attributes:
#         act (torch.nn.Module): The activation function applied after the convolution. Defaults to SiLU.
#         layer (GATv2Conv): The graph attention convolution layer.
#         dropout (torch.nn.Module): The dropout layer applied after the activation function to prevent overfitting.
#         norm (torch.nn.Module, optional): The normalization layer applied before activation.
#     """

#     def __init__(
#         self,
#         dims: int,
#         act: torch.nn.Module = SiLU,
#         dr: float = 0.1,
#         heads: int = 4,
#         edge_dim: Optional[int] = None,
#         norm: Optional[torch.nn.Module] = None,
#     ):
#         super().__init__()
#         self.act = act()
#         # We concatenate output from heads, so expect (dims * heads) dimensional inputs.
#         self.layer = GATv2Conv(dims * heads, dims, heads=heads, edge_dim=edge_dim)
#         self.dropout = torch.nn.Dropout(dr)
#         self.norm = norm(dims * heads) if norm is not None else None

#     def forward(
#         self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Forward pass through the residual layer.

#         Args:
#             x (torch.Tensor): The input node feature matrix with shape [num_nodes, dims * heads].
#             edge_index (torch.Tensor): The edge indices defining the graph structure.
#             edge_attr (torch.Tensor): The edge attributes, with shape [num_edges, edge_dim].

#         Returns:
#             torch.Tensor: The output node feature matrix with shape [num_nodes, dims].
#         """
#         h = self.layer(x, edge_index, edge_attr)

#         if self.norm is not None:
#             h = self.norm(h)

#         h = self.act(h)
#         return self.dropout(x + h)


class ResBlock(torch.nn.Module):
    """
    A stack of residual layers.

    Args:
        dims (int): The dimensionality of the input features (per head).
        heads (int): Number of attention heads.
        depth (int): Number of residual layers.

    Attributes:
        res_block (torch_geometric.nn.Sequential): sequence of resblock to pass input through
    """

    def __init__(
        self, dims: int, heads: int, depth: int, edge_dim: Optional[int] = None,
        add_self_loops: bool = True
    ):
        super().__init__()
        self.res_block = PyG_Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResLayer(dims, heads=heads, edge_dim=edge_dim,
                             add_self_loops=add_self_loops),
                    "x, edge_index, edge_attr -> x",
                )
                for _ in range(depth)
            ],
        )

    def forward(self, x: int, ei: torch.Tensor, ea: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): The input node feature matrix with shape [num_nodes, dims * heads].
            edge_index (torch.Tensor): The edge indices defining the graph structure.
            edge_attr (torch.Tensor): The edge attributes, with shape [num_edges, edge_dim].

        Returns:
            torch.Tensor: The output node feature matrix with shape [num_nodes, dims].
        """
        return self.res_block(x, ei, ea)
