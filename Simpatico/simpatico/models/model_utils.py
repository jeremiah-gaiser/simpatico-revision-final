import torch
from typing import Optional
from torch.nn import SiLU, Sequential, Linear
from torch_geometric.nn import GATv2Conv, knn, Sequential as PyG_Sequential
from torch_geometric.utils import to_undirected


class PositionalEdgeGenerator(torch.nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.MLP = Sequential(
            Linear(node_dim * 2 + 1, hidden_dim),
            self.relu,
            Linear(hidden_dim, 1),
            self.relu,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        x_subset: torch.Tensor,
        y_subset: torch.Tensor,
        k: int,
        batch: torch.Tensor,
    ):
        device = x.device
        connections = knn(
            pos[x_subset], pos[y_subset], k, batch[x_subset], batch[y_subset]
        )
        edge_index = torch.vstack(
            (y_subset[connections[0]], x_subset[connections[1]])
        ).to(device)
        node_distances = (
            torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            .unsqueeze(1)
            .to(device)
        )
        mlp_features = torch.hstack(
            (x[edge_index[0]], x[edge_index[1]], node_distances)
        )
        edge_weights = self.MLP(mlp_features)
        return to_undirected(edge_index, edge_weights)


class ResLayer(torch.nn.Module):
    """
    A residual layer that performs graph attention convolution (GATv2) and adds the result to the input.

    Attributes:
        act (torch.nn.Module): The activation function applied after the convolution. Defaults to SiLU.
        layer (GATv2Conv): The graph attention convolution layer.
        dropout (torch.nn.Module): The dropout layer applied after the activation function to prevent overfitting.
        norm (torch.nn.Module, optional): The normalization layer applied before activation.

    Args:
        dims (int): The dimensionality of the input features (per head).
        act (torch.nn.Module, optional): The activation function to use. Defaults to SiLU.
        dr (float, optional): The dropout rate. Defaults to 0.25.
        heads (int, optional): The number of attention heads in the GATv2 layer. Defaults to 4.
        edge_dim (int, optional): Number of dimensions in edge attributes.
        norm (torch.nn.Module, optional): The normalization layer applied before activation.
    """

    def __init__(
        self,
        dims: int,
        act: torch.nn.Module = SiLU,
        dr: float = 0.25,
        heads: int = 4,
        edge_dim: Optional[int] = None,
        norm: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.act = act()
        # We concatenate output from heads, so expect (dims * heads) dimensional inputs.
        self.layer = GATv2Conv(dims * heads, dims, heads=heads, edge_dim=edge_dim)
        self.dropout = torch.nn.Dropout(dr)
        self.norm = norm

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the residual layer.

        Args:
            x (torch.Tensor): The input node feature matrix with shape [num_nodes, dims * heads].
            edge_index (torch.Tensor): The edge indices defining the graph structure.
            edge_attr (torch.Tensor): The edge attributes, with shape [num_edges, edge_dim].

        Returns:
            torch.Tensor: The output node feature matrix with shape [num_nodes, dims].
        """
        h = self.layer(x, edge_index, edge_attr)

        if self.norm is not None:
            h = self.norm(h)

        h = self.act(h)
        return self.dropout(x + h)


class ResBlock(torch.nn.Module):
    """
    A stack of residual layers.

    Attributes:
        res_block (torch_geometric.nn.Sequential): sequence of resblock to pass input through

    Args:
        dims (int): The dimensionality of the input features (per head).
        heads (int): Number of attention heads.
        depth (int): Number of residual layers.

    Returns:
        torch.Tensor: The output node feature matrix with shape [num_nodes, dims].
    """

    def __init__(
        self, dims: int, heads: int, depth: int, edge_dim: Optional[int] = None
    ):
        super().__init__()
        self.res_block = PyG_Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResLayer(dims, heads=heads, edge_dim=edge_dim),
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
