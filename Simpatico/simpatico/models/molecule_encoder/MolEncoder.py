import torch
from simpatico.utils.model_utils import ResLayer, ResBlock
from torch_geometric.nn import (
    GATv2Conv,
    Sequential as PyG_Sequential,
)
from torch.nn import Sequential, Linear, ModuleList, SiLU
from typing import Optional
from simpatico.models import MolEncoderDefaults

class MolEncoder(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = MolEncoderDefaults["feature_dim"],
        hidden_dim: int = MolEncoderDefaults["hidden_dim"],
        out_dim: int = MolEncoderDefaults["out_dim"],
        heads: int = 4,
        blocks: int = 6,
        block_depth: int = 2,
        edge_dim: Optional[int] = MolEncoderDefaults["edge_dim"],
        max_hops: int = 3,
        **kwargs
    ):
        super().__init__()

        # Keep only edges up to `max_hops` covalent bonds. get_mol_edges bakes 1-, 2- and 3-hop
        # edges into every cached graph with a one-hot hop feature, so the restriction is a mask
        # on edge_attr rather than a rebuild of the corpus -- and edge_dim stays 3, so weights
        # remain compatible across settings.
        #
        # 12 GATv2 layers over 3-hop edges is a NOMINAL receptive field of 36 bonds, far wider
        # than any drug-like molecule, so in principle every atom sees the whole graph and can
        # encode which molecule it belongs to. Measured, the effective field is ~6-8 bonds, but
        # at max_hops=1 the nominal field drops to 12 bonds and the effective one well below
        # that, which is the point: an atom rep should describe a local chemical environment,
        # not identify its parent molecule.
        #
        # NOT in the state_dict -- see the note on ProteinEncoder.surface_only. Construct via
        # load_encoders(), which reads arch.json.
        self.max_hops = max_hops

        # Dimensions per head
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Initial projection
        self.input_projection_layer = torch.nn.Linear(feature_dim, hidden_dim * heads)

        # Backbone
        # Note: This uses the refactored ResBlock/ResLayer from our previous discussion
        # which now includes internal Pre-Activation LayerNorm.
        self.residual_blocks = torch.nn.ModuleList(
            [
                ResBlock(hidden_dim, heads, block_depth, edge_dim=edge_dim)
                for _ in range(blocks)
            ]
        )

        # Output Projection
        # Calculate size of the concatenated vector: (Initial + 6 blocks) * width
        concat_dim = (blocks + 1) * hidden_dim * heads

        self.output_projection_layer = torch.nn.Sequential(
            # CRITICAL: Normalize the massive concatenated vector
            torch.nn.LayerNorm(concat_dim),
            torch.nn.Linear(concat_dim, hidden_dim * heads),
            torch.nn.SiLU(), # Matches the SiLU in your ResLayers
            torch.nn.Dropout(0.1), # Add dropout before final classification
            torch.nn.Linear(hidden_dim * heads, out_dim),
        )

    def forward(self, data):
        """
        Forward method. Produces atom-level embeddings of PyG molecular graphs.
        """
        x, edge_index, edge_attr = (
            data.x.float(),
            data.edge_index.long(),
            data.edge_attr.float(),
        )

        if self.max_hops is not None and self.max_hops < edge_attr.size(1):
            # edge_attr is a one-hot of hop distance, and get_k_hop_edges never emits the same
            # atom pair at two hop counts, so a row is exactly one hop class.
            keep = edge_attr[:, : self.max_hops].sum(1) > 0.5
            edge_index, edge_attr = edge_index[:, keep], edge_attr[keep]

        # Initial projection
        # rblock_outs[0] shape: [N, hidden_dim * heads]
        rblock_outs = [self.input_projection_layer(x)]

        # Pass through Residual Blocks
        # The ResBlock handles the internal GATv2Conv and Residual addition
        for rblock in self.residual_blocks:
            rblock_outs.append(rblock(rblock_outs[-1], edge_index, edge_attr))

        # Aggregation
        # Concatenate features from all depths (DenseNet style)
        # axis=1 is the feature dimension
        concat_features = torch.cat(rblock_outs, dim=1)

        # Project to final embedding
        encoding = self.output_projection_layer(concat_features)
        return torch.nn.functional.normalize(encoding, p=2, dim=1)

# class MolEncoder(torch.nn.Module):
#     """
#     Generates molecular atom embeddings from molecular graph.
#     Args:
#         feature_dim (int, optional): Dimensionality of the input features.
#         hidden_dim (int, optional): Dimensionality of hidden layers.
#         out_dim (int, optional): Dimensionality of output.
#         heads (int, optional): Number of attention heads in graph attention layers (default 4).
#         blocks (int, optional): Number of residual blocks in model (default 3).
#         block_depth (int, optional): Number of residual layers in residual blocks (default 2).

#     Attributes:
#         input_projection_layer (torch.nn.Module): linear layer to project node features into [num_nodes, dims * heads]
#                                                   size tensor expected by residual block.
#         residual_blocks (torch.nn.ModuleList): List of residual blocks. Outputs will be concatenated in final layer.
#         output_projection_layer (torch.nn.Module): Non-linear layer which takes concatenation of residual_blocks outputs
#                                                     and outputs final embeddings.
#     """

#     def __init__(
#         self,
#         feature_dim: int = MolEncoderDefaults["feature_dim"],
#         hidden_dim: int = MolEncoderDefaults["hidden_dim"],
#         out_dim: int = MolEncoderDefaults["out_dim"],
#         heads: int = 4,
#         blocks: int = 6,
#         block_depth: int = 2,
#         edge_dim: Optional[int] = MolEncoderDefaults["edge_dim"],
#         **kwargs
#     ):
#         super().__init__()
#         self.input_projection_layer = Linear(feature_dim, hidden_dim * heads)

#         self.residual_blocks = ModuleList(
#             [
#                 ResBlock(hidden_dim, heads, block_depth, edge_dim=edge_dim)
#                 for _ in range(blocks)
#             ]
#         )

#         self.output_projection_layer = Sequential(
#             Linear((blocks + 1) * hidden_dim * heads, hidden_dim * heads),
#             SiLU(),
#             Linear(hidden_dim * heads, out_dim),
#         )

#     def forward(self, data):
#         """
#         Forward method. Produces atom-level embeddings of PyG molecular graphs.
#         Args:
#             data (Batch): PyG batch of small-molecule graphs.
#         Returns:
#             (torch.Tensor): (N, self.out_dim) shaped tensor of embedding values generated for each atom.
#         """
#         x, edge_index, edge_attr = (
#             data.x,
#             data.edge_index.long(),
#             data.edge_attr.float(),
#         )
#         # Outputs will be concatenated and input to final layer.
#         rblock_outs = [self.input_projection_layer(x.float())]

#         for rblock in self.residual_blocks:
#             rblock_outs.append(rblock(rblock_outs[-1], edge_index, edge_attr))

#         encoding = self.output_projection_layer(torch.hstack(rblock_outs))
#         return encoding
