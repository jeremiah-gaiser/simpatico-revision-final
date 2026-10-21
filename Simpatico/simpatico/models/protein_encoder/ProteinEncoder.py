import math
import sys
from copy import deepcopy

import torch
from torch.nn import Dropout, LayerNorm, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, Sequential, knn_graph
from torch_geometric.nn.aggr import AttentionalAggregation, MaxAggregation
from torch_geometric.nn.models import MLP
from torch_geometric.nn.pool import knn, radius
from torch_geometric.utils import dropout_edge, to_undirected

from simpatico.models import ProteinEncoderDefaults
from simpatico.utils.graph_utils import surface_mask
from simpatico.utils.model_utils import PositionalEdgeGenerator, ResBlock


class ProteinEncoder(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = ProteinEncoderDefaults["feature_dim"],
        hidden_dim: int = ProteinEncoderDefaults["hidden_dim"],
        out_dim: int = ProteinEncoderDefaults["out_dim"],
        heads: int = 4,
        blocks: int = 3,
        block_depth: int = 2,
        atom_k: int = 5,
        atom_vox_k: int = 5,
        vox_k: int = 5,
        add_self_loops: bool = True,
        surface_only: bool = False,
    ):

        super().__init__()
        # GAT layers concatenate heads, so true hidden dim is (hidden_dim*heads) dimensional.
        adjusted_hidden_dim = hidden_dim * heads

        # Nonlinear functions to pass edges through.
        self.atom_edge_MLP = PositionalEdgeGenerator(8)
        self.atom_vox_edge_MLP = PositionalEdgeGenerator(8)
        self.vox_edge_MLP = PositionalEdgeGenerator(8)

        # Nearest-neighbor count for atom-atom network
        self.atom_k = atom_k
        # Nearest-neighbor count for atom-voxel network
        self.atom_vox_k = atom_vox_k
        # Nearest-neighbor count for voxel-voxel network
        self.vox_k = vox_k

        # Restrict the whole pocket graph -- voxel nodes AND their atom context -- to
        # solvent-exposed atoms. A buried atom cannot contact the ligand, so the only thing
        # its presence can contribute to a voxel embedding is a unique key: measured on
        # ts005_th005, a voxel's own atom identity explains just 10% of its embedding
        # variance and a single-atom perturbation moves it by 0.0045, i.e. the embedding is
        # overwhelmingly a fingerprint of the surrounding atom CONFIGURATION. That is what a
        # hash key looks like. Surface atoms are well determined by residue and atom type,
        # so dropping the interior loses little real information and a lot of key material.
        # Roughly half the atoms and half the voxels survive (52% / 47% over 30 pockets).
        #
        # NOT a state_dict entry -- it changes no parameter shape, so a checkpoint trained
        # with it will silently load into a default encoder. train.py writes arch.json beside
        # the weights and load_encoders() reads it; construct through that, not directly.
        self.surface_only = surface_only

        self.atom_input_projection = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.vox_input_projection = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.residual_blocks = torch.nn.ModuleList(
                    [
                        ResBlock(
                            hidden_dim,
                            heads,
                            block_depth,
                            edge_dim=1,
                            add_self_loops=add_self_loops,
                            # You might need to update ResBlock to accept/pass 'norm' arg
                            # or hardcode LayerNorm inside ResLayer as done above
                        )
                        for _ in range(blocks)
                    ]
                )

        input_feat_size = (blocks + 1) * adjusted_hidden_dim

        self.output_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(input_feat_size), # ADDED: Normalize concatenated features
            torch.nn.Linear(input_feat_size, adjusted_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1), # ADDED: Dropout before final classification
            torch.nn.Linear(adjusted_hidden_dim, out_dim),
        )


    def forward(self, data, pocket_coords, pocket_radius=6):
        x, pos = (
            data.x.float(),
            data.pos
        )

        device = x.device

        if data.batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=device)
        else:
            batch = data.batch
        # Callers that hand in a single graph build this themselves and several do it as a
        # FLOAT tensor (rescore_litpcba.py, decoy_eval). Everything below uses it as an index,
        # and the surface_only branch scatters with it, which rejects anything but int64.
        if batch.dtype != torch.long:
            batch = batch.long()

        pocket_mask = torch.zeros(len(x)).bool().to(device)
        pocket_atom_index = radius(pocket_coords.pos, pos, pocket_radius, pocket_coords.batch, batch)[0].unique()
        pocket_mask[pocket_atom_index] = True

        if self.surface_only:
            # `data.surface` is precomputed once per protein by ProteinLigandDataLoader; the
            # fallback keeps single-graph inference paths working without a cache.
            surf = getattr(data, "surface", None)
            if surf is None:
                surf = surface_mask(pos, batch)
            surf = surf.to(device)
            masked = pocket_mask & surf
            # A graph left with no voxels would silently vanish from the output batch and shift
            # every later graph's index. Restore the unmasked pocket for any such graph.
            empty = torch.ones(int(batch.max()) + 1, dtype=torch.bool, device=device)
            empty.scatter_(0, batch[masked], False)
            pocket_mask = masked | (pocket_mask & empty[batch])

        # Trim atoms that are excessively far from the voxel nodes.
        trimmed_atom_index = radius(pos[pocket_mask], pos, 8, batch[pocket_mask], batch)[0].unique()
        if self.surface_only:
            trimmed_atom_index = trimmed_atom_index[surf[trimmed_atom_index]]

        atom_x = self.atom_input_projection(x[trimmed_atom_index])
        atom_pos = pos[trimmed_atom_index]
        atom_batch = batch[trimmed_atom_index]

        vox_x = self.vox_input_projection(x[pocket_mask])
        vox_pos = pos[pocket_mask]
        vox_batch = batch[pocket_mask]

        full_x = torch.vstack((atom_x, vox_x))
        full_pos = torch.vstack((atom_pos, vox_pos))
        full_batch = torch.hstack((atom_batch, vox_batch))

        atom_node_index = torch.arange(atom_x.size(0)).to(device)
        vox_node_index = (torch.arange(vox_x.size(0)) + atom_node_index.size(0)).to(
            device
        )

        edge_combos = [
            (self.atom_edge_MLP, [atom_node_index, atom_node_index, self.atom_k]),
            (
                self.atom_vox_edge_MLP,
                [vox_node_index, atom_node_index, self.atom_vox_k],
            ),
            (self.vox_edge_MLP, [vox_node_index, vox_node_index, self.vox_k]),
        ]

        aa_edges, av_edges, vv_edges = [
            f(full_pos, *edge_params, full_batch) for f, edge_params in edge_combos
        ]

        full_edge_index = torch.hstack((aa_edges[0], av_edges[0], vv_edges[0])).to(
            device
        )
        full_edge_attr = torch.vstack((aa_edges[1], av_edges[1], vv_edges[1])).to(
            device
        )

        # ==================================================================
        #  DROPEDGE IMPLEMENTATION
        # ==================================================================
        if self.training:
            # p=0.2 means 20% of edges are removed randomly
            full_edge_index, edge_mask = dropout_edge(
                edge_index=full_edge_index,
                p=0.2,
                force_undirected=False,
                training=True
            )
            full_edge_attr = full_edge_attr[edge_mask]
        # ==================================================================

        rblock_outs = [full_x]

        for rblock in self.residual_blocks:
            rblock_outs.append(rblock(rblock_outs[-1], full_edge_index, full_edge_attr))

        raw = self.output_projection(torch.hstack(rblock_outs)[vox_node_index])
        encoding = torch.nn.functional.normalize(raw, p=2, dim=1)

        # vox_index maps each emitted voxel back to its row in the input graph. build_pair_index
        # needs it: it cannot re-derive the pocket from a radius() call once surface_only is on.
        # x_raw is the activation BEFORE F.normalize. Training-only, for SIGReg, which targets
        # isotropic N(0, I) and so cannot act on the unit-sphere output -- see train.sigreg.
        # Carried on the Data object rather than via a forward hook so it survives the
        # no_grad/eval paths unchanged and costs nothing when unused.
        return Data(x=encoding, x_raw=raw, pos=vox_pos, batch=vox_batch,
                    vox_index=pocket_mask.nonzero(as_tuple=True)[0])
