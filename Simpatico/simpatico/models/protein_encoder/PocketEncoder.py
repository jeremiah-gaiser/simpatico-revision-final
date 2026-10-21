import torch
from torch_geometric.data import Data

from simpatico.models import ProteinEncoderDefaults
from simpatico.utils.model_utils import PositionalEdgeGenerator, ResBlock


class PocketEncoder(torch.nn.Module):
    """One node population, one projection, one edge set.

    ProteinEncoder builds a two-population graph: `voxel` nodes (pocket atoms, which emit the
    output) and `atom` context nodes (everything within 8 A of them), each with its own input
    projection, joined by three separate edge generators (atom-atom, voxel-atom, voxel-voxel).
    Every pocket atom therefore enters the graph TWICE, once in each role. Measured over 12
    complexes with surface_only on: 54 voxel nodes and 208 context nodes per complex, 262 node
    slots of which 54 are duplicates, and ~79% of the graph is context.

    That design is a leftover from a more expensive pocket representation. Here the graph is:

        nodes = protein atoms within `pocket_radius` of ANY ligand atom, no surface filter
        edges = one radius graph over those nodes, one PositionalEdgeGenerator
        output = one embedding per node

    ~110 nodes per complex (the 6 A shell before filtering), against 262 slots today.

    The surface filter is deliberately gone, not merely defaulted off. It was discarding 31%
    of the interaction signal: of protein atoms within the 4 A interaction threshold, only
    69.0% are solvent-exposed (400 complexes, 11,102/16,090 pooled; per-complex mean 0.704,
    sd 0.127, bottom decile 0.55). Because contrastive_loss derives positives from the
    ENCODER'S OUTPUT rows, a buried contact atom was not down-weighted but absent, and could
    never be a positive. 4.7% of complexes were losing more than half their contacts.

    Parameter shapes differ from ProteinEncoder (one projection, one edge MLP), so a
    checkpoint from one fails load_state_dict into the other loudly rather than silently
    evaluating a different model. load_encoders() dispatches on arch.json's "kind".
    """

    def __init__(
        self,
        feature_dim: int = ProteinEncoderDefaults["feature_dim"],
        hidden_dim: int = ProteinEncoderDefaults["hidden_dim"],
        out_dim: int = ProteinEncoderDefaults["out_dim"],
        heads: int = 4,
        blocks: int = 3,
        block_depth: int = 2,
        edge_radius: float = 5.0,
        pocket_radius: float = 6.0,
        drop_edge: float = 0.2,
        add_self_loops: bool = True,
    ):
        super().__init__()
        # GAT layers concatenate heads, so true hidden dim is (hidden_dim*heads) dimensional.
        adjusted_hidden_dim = hidden_dim * heads

        # Named `edge_radius` rather than the `*_k` of ProteinEncoder, whose atom_k/vox_k are
        # commented as neighbour COUNTS but are passed to PositionalEdgeGenerator as the
        # radius argument of torch_geometric radius() -- 5 A capped at 32 neighbours, not k=5.
        self.edge_radius = edge_radius
        # Pocket definition: atoms within this of any ligand atom. Carried on the module so
        # arch.json records it; a checkpoint evaluated at a different pocket radius sees a
        # different graph and no parameter shape would catch it.
        self.pocket_radius = pocket_radius
        self.drop_edge = drop_edge

        self.input_projection = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.edge_MLP = PositionalEdgeGenerator(8)

        self.residual_blocks = torch.nn.ModuleList(
            [
                ResBlock(hidden_dim, heads, block_depth, edge_dim=1,
                         add_self_loops=add_self_loops)
                for _ in range(blocks)
            ]
        )

        input_feat_size = (blocks + 1) * adjusted_hidden_dim
        self.output_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(input_feat_size),
            torch.nn.Linear(input_feat_size, adjusted_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(adjusted_hidden_dim, out_dim),
        )

    def forward(self, data, pocket_coords, pocket_radius=None):
        from torch_geometric.nn.pool import radius
        from torch_geometric.utils import dropout_edge

        x, pos = data.x.float(), data.pos
        device = x.device
        r = self.pocket_radius if pocket_radius is None else pocket_radius

        batch = data.batch
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=device)
        # Several single-graph callers (rescore_litpcba.py, decoy_eval) build this as a float
        # tensor; everything below indexes with it.
        elif batch.dtype != torch.long:
            batch = batch.long()

        # radius(x, y, r)[0] indexes y, so this is protein atoms within r of ANY ligand atom.
        # .unique() returns them sorted, which build_pair_index relies on: it maps graph rows
        # back through node_index and requires ascending order aligned with the output rows.
        node_index = radius(
            pocket_coords.pos, pos, r, pocket_coords.batch, batch
        )[0].unique()

        # A graph with no nodes would vanish from the output batch and shift every later
        # graph's index. Without the surface filter this needs a ligand with no protein atom
        # within r, which does not occur in PDBbind (0/400 complexes had zero contacts even at
        # the tighter 4 A threshold), so fail loudly rather than silently renumber.
        counts = torch.bincount(batch[node_index], minlength=int(batch.max()) + 1)
        if not bool((counts > 0).all()):
            raise RuntimeError(
                f"{int((counts == 0).sum())} graph(s) have no protein atom within {r} A of "
                f"their ligand; output rows would silently misalign with the batch"
            )

        node_x = self.input_projection(x[node_index])
        node_pos = pos[node_index]
        node_batch = batch[node_index]

        # One edge set over one population. x_subset and y_subset are the same index range
        # because the generator takes subsets of the tensor it is handed, and it is handed the
        # already-subset node positions.
        all_nodes = torch.arange(node_x.size(0), device=device)
        edge_index, edge_attr = self.edge_MLP(
            node_pos, all_nodes, all_nodes, self.edge_radius, node_batch
        )

        if self.training and self.drop_edge:
            edge_index, edge_mask = dropout_edge(
                edge_index=edge_index, p=self.drop_edge,
                force_undirected=False, training=True,
            )
            edge_attr = edge_attr[edge_mask]

        block_outs = [node_x]
        for rblock in self.residual_blocks:
            block_outs.append(rblock(block_outs[-1], edge_index, edge_attr))

        raw = self.output_projection(torch.hstack(block_outs))
        encoding = torch.nn.functional.normalize(raw, p=2, dim=1)

        # `vox_index` keeps ProteinEncoder's field name: build_pair_index and any other
        # consumer reads it by that name to map output rows back to input graph rows.
        # x_raw is pre-normalisation, for SIGReg, which targets isotropic N(0, I) and so
        # cannot act on the unit-sphere output.
        return Data(x=encoding, x_raw=raw, pos=node_pos, batch=node_batch,
                    vox_index=node_index)
