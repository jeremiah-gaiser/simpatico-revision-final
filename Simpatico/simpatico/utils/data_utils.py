import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn, radius

from simpatico.utils import graph_utils
from simpatico.utils.utils import get_k_hop_edges


def concatenate_pyg_files(file_list):
    pyg_graphs = []

    for f in file_list:
        g = torch.load(f, weights_only=False)
        if isinstance(g, Batch):
            pyg_graphs = Batch.to_data_list(g)
        else:
            pyg_graphs.append(g)

    pyg_batch = Batch.from_data_list(pyg_graphs)
    return pyg_batch


def handle_no_overwrite(outfile: str) -> bool:
    """
    If file does not exist, generate a sentinel file and return True.
    Otherwise return False
    Args:
        outfile (str): file path
    Returns:
        (bool): Truth indicates that output file did not exists.
    """
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        return False
    else:
        # Ensure that parallel jobs know this file is being worked on elsewhere.
        Path(outfile).touch()
        return True


def report_results(query_output: Tuple) -> None:
    """
    Print human-readable results from vector database query/screen.
    Args:
        query_output (Tuple): query indexes, database indexes, and the list of sorted per-item results.
    Returns:
        (None): prints output
    """
    queries, vector_db, results = query_output
    target_list = ">query sources:\n"
    db_list = ">db sources:\n"
    results_content = ">results:\n"

    for t_i, file_string in enumerate(queries["sources"]):
        target_list += f"{t_i+1} {file_string}\n"

    for d_i, file_string in enumerate(vector_db["sources"]):
        db_list += f"{d_i+1} {file_string}\n"

    for qi, result_vals in enumerate(results):
        rank = 1
        q_item_index = queries["item_batch"] == qi
        q_source_idx = queries["file_batch"][q_item_index][0].item()
        q_source_item_idx = queries["source_index"][q_item_index][0].item()

        for result_idx in result_vals[1]:
            r_idx = result_idx.item()
            r_item_index = vector_db["item_batch"] == r_idx
            r_source_idx = vector_db["file_batch"][r_item_index][0].item()
            r_source_item_idx = vector_db["source_index"][r_item_index][0].item()

            results_content += f"{q_source_idx+1} {q_source_item_idx+1} {r_source_idx+1} {r_source_item_idx+1} {rank}\n"
            rank += 1

    output = target_list + db_list + results_content
    print(output)


class ProteinLigandDataLoader:
    """
    Data loader for storing and producing protein-ligand pairs during training.

    Args:
        pl_graph_pairs (List[(Data, Data)]): List of protein-ligand graph pairs.
        batch_size (int): batch size

    Attributes:
        proteins (List[Data]): N-length list of proteins
        ligands (List[Data]): N-length list of ligands
        batch_size (int): batch size
    """

    def __init__(self, pl_graph_pairs: List[Data], batch_size: int,
                 negative_pool: Optional[List[Data]] = None,
                 negative_mix: float = 0.0,
                 surface: bool = False,
                 surface_cache: Optional[str] = None):
        self.proteins = []
        self.ligands = []
        self.batch_size = batch_size
        # Hard negatives are drawn from `negative_pool` when supplied, otherwise from the
        # training ligands themselves. The default means the model learns to reject other
        # CRYSTAL ligands, while at screening time every negative is a drug-like library
        # compound -- a distribution mismatch that measurably matters: swapping the validation
        # pool from crystal ligands to ZINC took the correlation between probe rank and
        # LIT-PCBA EF from -0.69 to -0.82.
        self.negative_pool = negative_pool
        # Fraction of each negative batch drawn from `negative_pool` rather than from the
        # crystal ligands. A ZINC-only pool (mix 1.0) flattened the library -- ALDH1 AUROC
        # 0.482, i.e. chance -- because InfoNCE then had nothing to keep true binders apart
        # from each other. Mixing keeps both claims in the objective: reject library compounds,
        # and still distinguish one crystal binder from another.
        self.negative_mix = 0.0 if negative_pool is None else float(negative_mix)

        for p_graph, l_graph in pl_graph_pairs:
            self.proteins.append(p_graph)
            self.ligands.append(l_graph)

        self.size = len(self.proteins)
        # Entries drawn per epoch. Equals `size` unless a pocket cap is set, which trims the
        # over-represented sites -- see set_pocket_caps.
        self.epoch_size = self.size
        self.pocket_cap = 0
        self.pocket_groups = None
        self.batch_iterator = self.new_batch_iterator()
        self.set_proximal_atom_masks()
        if surface:
            self.set_surface_masks(surface_cache)

    def set_surface_masks(self, cache_file: Optional[str] = None):
        """Attach `surface` (per-atom boolean) to every protein graph.

        Computed once here rather than inside ProteinEncoder.forward: the mask depends only on
        coordinates, so it is constant across epochs, and running Shrake-Rupley every step
        would add ~0.4 s to a batch of 16. The 0.25 A coordinate jitter training_step applies
        is far below the exposed/buried decision boundary, so the precomputed mask stays valid.

        Cached to `cache_file` keyed by graph name, so a restart pays nothing.
        """
        cache = {}
        if cache_file and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            cache = torch.load(cache_file, weights_only=False)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fresh = 0
        for g in self.proteins:
            key = getattr(g, "name", None)
            m = cache.get(key) if key is not None else None
            if m is None or m.numel() != g.pos.size(0):
                m = graph_utils.surface_mask(g.pos.to(dev).float()).cpu()
                fresh += 1
                if key is not None:
                    cache[key] = m
            g.surface = m
        if cache_file and fresh:
            torch.save(cache, cache_file)

    def get_random_ligand_batch(self, n, blacklist=None):
        """A batch of negatives: `negative_mix` of it from `negative_pool`, the rest crystal.

        The pool draw is uniform. The pool itself must also be a uniform sample of its source
        library -- the MaxMin-diverse libraries under dock_library/ optimise for spread and so
        over-represent the edges of chemical space, which is the right objective for a
        SCREENING library and the wrong one for a negative distribution meant to look like what
        the model will be scored against.
        """
        n_pool = int(round(n * self.negative_mix)) if self.negative_pool is not None else 0
        mol_batch = []
        if n_pool:
            idx = torch.randint(0, len(self.negative_pool), (n_pool,)).tolist()
            mol_batch += [self.negative_pool[i] for i in idx]

        for idx in torch.randperm(len(self.ligands)).tolist():
            if len(mol_batch) >= n:
                break
            lig = self.ligands[idx]
            if blacklist is not None and lig.ligand_id in blacklist:
                continue
            mol_batch.append(lig)
        return Batch.from_data_list(mol_batch)




    def set_pocket_ids(self, pocket_file: str):
        """Attach a pocket-group id to every protein graph, so a batch knows which of its own
        entries are the SAME binding site.

        Needed by lig_external_pool="protein_inbatch", whose negatives are the other complexes
        in the batch. Masking the anchor's own complex is not enough: PDBbind solves popular
        sites repeatedly, so at batch 64 roughly 10% of anchors share a pocket with some other
        entry in the same batch (3.4 colliding pairs per batch; pairwise probability 0.00166
        over 6,406 pockets / 19,434 entries). Those are not negatives, they are the anchor's
        own site under another entry, they rank among the hardest, and at t_hard 0.05 the
        softmax puts its mass exactly there.

        Same identity source as set_pocket_caps -- MMseqs2 95%/0.8 on chains then split by
        pocket-residue Jaccard >= 0.5. Entries absent from the map get their own singleton id,
        the conservative reading: never masked against anything but themselves.
        """
        import json

        with open(pocket_file) as f:
            pk = json.load(f)
        ent2pocket = {}
        for pid, grp in enumerate(pk["pockets"]):
            for e in grp[0]:
                ent2pocket[e] = pid

        next_id = len(pk["pockets"])
        unmapped = 0
        for i, prot in enumerate(self.proteins):
            eid = str(getattr(prot, "name", i)).split("_")[0]
            q = ent2pocket.get(eid)
            if q is None:
                q, next_id, unmapped = next_id, next_id + 1, unmapped + 1
            prot.pocket_gid = torch.tensor([q], dtype=torch.long)
        return {"pockets": len(pk["pockets"]), "unmapped": unmapped}

    def set_pocket_caps(self, pocket_file: str, cap: int):
        """Allow at most `cap` entries from any one pocket per epoch, resampled every epoch.

        PDBbind solves popular targets repeatedly, so uniform sampling over entries spends
        gradient in proportion to how often a site was crystallised. On the litpcba_s90 split
        that is severe: 18,948 entries cover 6,262 pockets, 31 pockets (0.5%) take 16.6% of
        every epoch, the largest is drawn 412 times, and the inverse-Simpson effective pocket
        count is 562 -- 9% of the nominal 6,262.

        A cap rather than inverse-frequency weights, because the redundancy is in the POCKET
        and not the ligand: those 31 pockets carry 2,811 distinct ligands across 3,137 entries,
        89.6% unique. Permanently downweighting them would discard real ligand diversity.
        Re-drawing which `cap` entries appear each epoch bounds per-epoch exposure while still
        walking the whole ligand set over a run -- at cap 10 the 412-entry pocket is fully
        covered in 42 epochs and never exceeds ~14 draws in any one.

        Pocket identity comes from pocket_census/pockets_j0.5.json: MMseqs2 95% identity / 0.8
        coverage on chains, then split by pocket-residue Jaccard >= 0.5 so an active site and
        an allosteric site on one protein count separately. Entries absent from the map get
        their own singleton pocket, which is the conservative reading (never over-capped).

        SET `epochs` BY STEP BUDGET, NOT BY EPOCH COUNT. Capping shortens the epoch, so a run
        compared at equal EPOCHS against an uncapped one silently gets a smaller training
        budget and the cap's effect is confounded with undertraining. On the litpcba_s90 split:

            cap    epoch entries   % of full   steps/epoch @ bs 64   epochs for 14,800 steps
            none          18,948      100.0%                   296                        50
              25          15,961       84.2%                   249                        59
              10          13,432       70.9%                   209                        71
               5          11,189       59.1%                   174                        85

        This sets `epoch_size`, which train.py's epoch loop must use -- `size` would
        over-request batches and silently restart the iterator part-way through an epoch.
        """
        import json
        from collections import defaultdict

        with open(pocket_file) as f:
            pk = json.load(f)
        ent2pocket = {}
        for pid, grp in enumerate(pk["pockets"]):
            for e in grp[0]:
                ent2pocket[e] = pid

        groups = defaultdict(list)
        unmapped = 0
        for i, prot in enumerate(self.proteins):
            eid = str(getattr(prot, "name", i)).split("_")[0]
            q = ent2pocket.get(eid)
            if q is None:
                q = ("UNMAPPED", eid)
                unmapped += 1
            groups[q].append(i)

        self.pocket_groups = list(groups.values())
        self.pocket_cap = int(cap)
        self.epoch_size = sum(min(len(g), self.pocket_cap) for g in self.pocket_groups)
        self.batch_iterator = self.new_batch_iterator()
        return {"pockets": len(self.pocket_groups), "unmapped": unmapped,
                "epoch_size": self.epoch_size, "full_size": self.size,
                "largest_pocket": max(len(g) for g in self.pocket_groups)}

    def new_batch_iterator(self):
        """
        Generator for random indices in upcoming batches.

        Yields:
            (torch.Tensor): A 1D tensor containing indices for the current batch.
        """
        if self.pocket_cap and self.pocket_groups is not None:
            # Fresh draw every epoch, so the capped slots rotate through each pocket's entries.
            keep = []
            for g in self.pocket_groups:
                if len(g) <= self.pocket_cap:
                    keep.extend(g)
                else:
                    sel = torch.randperm(len(g))[: self.pocket_cap].tolist()
                    keep.extend(g[i] for i in sel)
            random_index = torch.tensor(keep, dtype=torch.long)
            random_index = random_index[torch.randperm(len(random_index))]
        else:
            random_index = torch.randperm(self.size)
        for i in range(0, len(random_index), self.batch_size):
            yield random_index[i : i + self.batch_size]

    def set_proximal_atom_masks(self, r: Optional[int] = 4):
        """
        For each protein-ligand pair, add per-node boolean masks to graphs in self.proteins and self.ligands to indicate atoms involved in interactions.
        Args:
            r (int, optional): radius value that determines interaction-distance
        """
        for g_i in range(self.size):
            protein_mask, ligand_mask = graph_utils.get_proximal_atom_masks(
                self.proteins[g_i].pos, self.ligands[g_i].pos, r
            )
            self.proteins[g_i].proximal = protein_mask
            self.ligands[g_i].proximal = ligand_mask

    def get_random_negatives(self, mol_batch):
        """
        returns random index of all atoms in molecule batch for random negatives.
        Args:
            mol_batch (Batch): molecular graph batch
        Returns:
            (torch.Tensor): random index
        """
        return torch.randperm(mol_batch.x.size(0))

    def get_batch(
        self,
        graph_index: torch.Tensor,
        skip_degenerate: bool = True,
    ):
        """
        Get batch of protein graphs and corresponding batch of ligand graphs based on index.

        The pocket itself is not set here: `ProteinEncoder.forward` derives it from the
        ligand coordinates it is handed, so training and inference share one definition.

        Args:
            graph_index (torch.Tensor): index of protein-ligand pairs.
            skip_degenerate (bool, optional): if True, drop pairs with no protein atom
                within the interaction radius of the ligand (see `set_proximal_atom_masks`).
                Such pairs have no pocket for the encoder to voxelise.

        Returns:
            (Batch, Batch): PyG batches of protein graphs and corresponding ligand graphs
        """

        protein_list = []
        mol_list = []

        for g_idx in graph_index:
            g_i = g_idx.item()

            if skip_degenerate and not self.proteins[g_i].proximal.any():
                continue

            protein_list.append(self.proteins[g_i].clone())
            mol_list.append(self.ligands[g_i].clone())

        protein_batch = Batch.from_data_list(protein_list)
        mol_batch = Batch.from_data_list(mol_list)

        return protein_batch, mol_batch

    def get_random_batch(self, skip_degenerate: bool = True) -> Batch:
        """
        Get a random batch of protein graphs and corresponding batch of ligand graphs.
        Args:
            skip_degenerate (bool, optional): see `get_batch`.

        Returns:
            (Batch, Batch): PyG batches of protein graphs and corresponding ligand graphs
        """
        random_index = next(self.batch_iterator, None)

        if random_index is None or len(random_index) == 1:
            self.batch_iterator = self.new_batch_iterator()
            random_index = next(self.batch_iterator)

        return self.get_batch(random_index, skip_degenerate)


class TrainingOutputHandler:
    """
    Organizes the output of protein and molecule encoders into inputs for the contrastive loss module during training.
    This entails selecting anchor-positive pairs and anchor-(random/self/hard) negative pairs.

    Args:
        protein_embeds (torch.Tensor)
        protein_pos (torch.Tensor)
        protein_batch (torch.Tensor)
        mol_embeds (torch.Tensor)
        mol_pos (torch.Tensor)
        mol_batch (torch.Tensor)
        interaction_radius (int, optional): threshold distance to define atomic interaction (default 4)

    Attributes:
        protein_embeds (torch.Tensor)
        protein_pos (torch.Tensor)
        protein_batch (torch.Tensor)
        mol_embeds (torch.Tensor)
        mol_pos (torch.Tensor)
        mol_batch (torch.Tensor)
        mol_actives (torch.Tensor): index of molecular atoms involved in interaction with corresponding item in self.prot_actives
        prot_actives (torch.Tensor): index of protein atoms involved in interaction with corresponding item in self.mol_actives
    """

    def __init__(
        self,
        protein_embeds: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        mol_embeds: torch.Tensor,
        mol_pos: torch.Tensor,
        mol_batch: torch.Tensor,
        interaction_radius: int = 4,
    ):
        self.protein_embeds = protein_embeds
        self.protein_pos = protein_pos
        self.protein_batch = protein_batch

        self.mol_embeds = mol_embeds
        self.mol_pos = mol_pos
        self.mol_batch = mol_batch

        self.mol_actives, self.prot_actives = radius(
            protein_pos, mol_pos, 4, protein_batch, mol_batch
        )

    def get_contact_matrix(self):
        mol_actives = self.mol_actives.unique()
        prot_actives = self.prot_actives.unique()

        mol_contacts, prot_contacts = radius(
            self.protein_pos[prot_actives],
            self.mol_pos[mol_actives], 4,
            self.protein_batch[prot_actives],
            self.mol_batch[mol_actives]
        )

        contact_map = torch.zeros((len(prot_actives), len(mol_actives)))
        contact_map[(prot_contacts, mol_contacts)] = 1

        # prot_norm = F.normalize(self.protein_embeds[prot_actives], p=2, dim=1)  # shape (N, D)
        # mol_norm = F.normalize(self.mol_embeds[mol_actives], p=2, dim=1)  # shape (M, D)

        inner_prod_matrix = self.protein_embeds[prot_actives] @ self.mol_embeds[mol_actives].T

        return inner_prod_matrix, contact_map.to(inner_prod_matrix.device)

    def get_mol_data(self, active_only=False):
        """
        Fetches encoding, positional and batch data of molecular embeddings.
        Args:
            active_only (bool): If true, only fetch values corresponding to active atom embeds (used as contrastive learning anchors).
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): encoding, positional and batch data.
        """
        if active_only:
            mol_index = self.mol_actives
        else:
            mol_index = torch.arange(self.mol_embeds.size(0))

        return (
            self.mol_embeds[mol_index],
            self.mol_pos[mol_index],
            self.mol_batch[mol_index],
        )

    def get_protein_data(self, active_only=False):
        """
        Fetches encoding, positional and batch data of protein embeddings.
        Args:
            active_only (bool): If true, only fetch values corresponding to active atom embeds (used as contrastive learning anchors).
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): encoding, positional and batch data.
        """
        if active_only:
            protein_index = self.prot_actives
        else:
            protein_index = torch.arange(self.protein_embeds.size(0))

        return (
            self.protein_embeds[protein_index],
            self.protein_pos[protein_index],
            self.protein_batch[protein_index],
        )

    def get_hard_negatives(self, prot_anchor=True, difficulty=0.25):
        """
        Fetches embeddings from non-corresponding structures with upper-quartile embedding similarity.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
            difficulty (float, optional): value between 0 and 1 specifying the quartile threshold
                                          (e.g. diffulty=0.1, negatives will be selected from 10% nearest non-positive embeddings)
        Returns:
            (torch.Tensor): index of hard negative embeds.
        """
        anchor_embeds, _, anchor_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        negative_embeds, _, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        embed_distances = torch.cdist(anchor_embeds, negative_embeds)
        k = int(embed_distances.size(1) * difficulty)

        self_mask = anchor_batch.unsqueeze(1) == negative_batch.unsqueeze(0)
        # Set the distance between any protein atom and true ligand embed to maximum observed distance
        # This prevents positive embeds from being selected
        embed_distances[self_mask] = embed_distances.max()

        # randomly select a negative from among the top-k closest embeds per anchor atom
        hard_negative_index = embed_distances.argsort(1)[:, :k][
            (
                torch.arange(embed_distances.size(0)),
                torch.randint(0, k, (embed_distances.size(0),)),
            )
        ]
        return hard_negative_index

    def get_self_negatives(self, prot_anchor=True):
        """
        Fetches non-interacting embeddings from corresponding structures.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
        Returns:
            (torch.Tensor): self negatives index
        """

        _, anchor_pos, anchor_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        _, negative_pos, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        negative_index = []
        self_mask = anchor_batch.unsqueeze(1) == negative_batch.unsqueeze(0)

        for a_pos, row in zip(anchor_pos, self_mask):
            # indices of negative embeds from same complex
            self_samples = torch.where(row)[0]

            spatial_distances = torch.cdist(
                a_pos.unsqueeze(0), negative_pos[self_samples]
            )[0]
            k = int(self_samples.size(0) * 0.75)

            # sort indices of negative embeds by decreasing spatial distance from anchor embed.
            # Cut the quarter nearest embeds.
            farthest_self_negatives = self_samples[
                spatial_distances.argsort(descending=True)
            ][:k]

            # Append one of these random 3/4 farthest self-negatives to 'negative_index'
            negative_index.append(
                farthest_self_negatives[
                    torch.randperm(farthest_self_negatives.size(0))[0]
                ]
            )
        return torch.hstack(negative_index)

    def get_random_negatives(self, prot_anchor=True):
        """
        Fetches random embeddings from non-corresponding structures.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
        Returns:
            (torch.Tensor): random negatives index
        """

        _, _, positive_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        _, _, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        random_index = []
        self_mask = positive_batch.unsqueeze(1) != negative_batch.unsqueeze(0)

        for row in self_mask:
            non_self_indices = torch.where(row)[0]
            random_idx = torch.randint(0, len(non_self_indices), (1,))
            random_index.append(non_self_indices[random_idx])

        return torch.hstack(random_index)

    def get_anchors_positives_negatives(self, prot_anchor=True, difficulty=0.25):
        """
        Assembles protein-anchor-negative embed collections to pass to contrastive loss module.
        Args:
            difficulty (float, optional): value between 0 and 1 specifying the quartile threshold
                                          (e.g. diffulty=0.1, negatives will be selected from 10% nearest non-positive embeddings)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Two (N, embed_dim) tensors of anchors and positives.
                                                        One (3*N, embed_dim) tensor of random, self, and hard negatives.
        """
        anchor_embeds = self.protein_embeds if prot_anchor else self.mol_embeds
        anchor_actives = self.prot_actives if prot_anchor else self.mol_actives
        pos_neg_embeds = self.mol_embeds if prot_anchor else self.protein_embeds
        pos_neg_actives = self.mol_actives if prot_anchor else self.prot_actives

        with torch.no_grad():
            random_negatives = self.get_random_negatives(prot_anchor=prot_anchor)
            self_negatives = self.get_self_negatives(prot_anchor=prot_anchor)
            # hard_negatives = self.get_hard_negatives(
            #     prot_anchor=prot_anchor, difficulty=difficulty
            # )

        all_negatives = torch.hstack((random_negatives, self_negatives))

        return (
            anchor_embeds[anchor_actives],
            pos_neg_embeds[pos_neg_actives],
            pos_neg_embeds[all_negatives],
        )
