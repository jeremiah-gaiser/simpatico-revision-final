import sys

import faiss
import numpy as np
import torch
from torch_geometric.data import Batch, Data


def cos_distance(sim):
    """Cosine similarity -> a distance on [0, 2]: 0 identical, 1 orthogonal, 2 antipodal.

    The scoring arithmetic in get_scores was written against squared L2 and is left
    untouched, because on unit vectors 1 - cos is EXACTLY half the squared L2 distance
    (d^2 = 2 - 2cos). Every quantity derived from it -- max_D, the q-quantile threshold, the
    hinge, the summed per-molecule score -- therefore just halves, and no ranking moves.
    What the switch buys is that the metric is now cosine by construction rather than by an
    invariant (encoder output being unit-norm) that lives in another module.
    """
    return 1.0 - sim


def apply_duplicate_mask(S, mol_I, batch_size=500):
    """
    Identifies duplicate IDs within each row of mol_I and zeros out
    the corresponding entries in S, excluding the first occurrence.

    Args:
        S (torch.Tensor): The target tensor to mask (e.g., shape [10647, 2048])
        mol_I (torch.Tensor): The ID tensor (shape [10647, 2048])
        batch_size (int): Number of rows to process at once to save VRAM.
    """
    num_rows = mol_I.size(0)
    # Initialize a 2D mask on the same device as mol_I
    duplicate_mask = torch.zeros_like(mol_I, dtype=torch.bool, device=mol_I.device)

    for i in range(0, num_rows, batch_size):
        end_i = min(i + batch_size, num_rows)
        chunk = mol_I[i:end_i]  # Shape: [batch, 2048]

        # 1. Perform comparison within the chunk
        # Shape: [batch, 2048, 2048]
        matches = chunk.unsqueeze(2) == chunk.unsqueeze(1)

        # 2. Get lower triangle (diagonal=-1 excludes comparing an element to itself)
        # This identifies if an element has appeared at any index PREVIOUS to it.
        previous_matches = torch.tril(matches, diagonal=-1)

        # 3. Collapse to 2D: True if the element at this position appeared earlier
        # Shape: [batch, 2048]
        duplicate_mask[i:end_i] = previous_matches.any(dim=2)

    # Apply the mask to S
    S[duplicate_mask] = 0
    return S


class VectorDatabase:
    """
    Interface for embedding collections to be used for building and querying FAISS vector databases.

    Args:
        embed_files (list[str]): list of filepaths to PyG graph files containing embedding values to store in VectorDatabase.
        score_threshold (tuple): max-distance and n-quantile distantance statistics used in aggregation/scoring function during query.

    Attributes:
        sources (list): ordered list of filepaths of ultimate vector sources.
        vectors (torch.Tensor): list of vectors.
        item_batch (torch.Tensor): pyg-style batch tensor corresponding vectors to a specific item/graph.
        file_batch (torch.Tensor): pyg-style batch tensor corresponding vectors to a source file.
        source_index (torch.Tensor): pyg-style batch tensor specifying which item of the original source file vector belongs to.
        score-thresholds (tuple): max-distance and n-quantile distantance statistics used in aggregation/scoring function during query.

    Distances throughout are 1 - cos (see cos_distance), which is HALF the squared L2 this
    class used previously. Rankings are unaffected, but a score_thresholds pickle saved by
    the old code is on the doubled scale -- reusing one via --load-thresholds would apply a
    2x bias and is not supported. Regenerate it.
    """

    def __init__(self, embed_batch, score_thresholds=None):
        self.score_thresholds = score_thresholds
        self.vectors = embed_batch.x
        self.batch = embed_batch.batch

    def get_score_thresholds(
        self, vector_db, n_random: int = 4000, q: float = 0.99, n_trials=5
    ):
        """
        Retrieve and store in `self.score_thresholds` per-item maximum observed distance and nth-quantile score values
        from a random sampling of distances between `self.vectors` and those of a different VectorDatabase
        (used for score aggregation during query).

        Args:
            vector_db (VectorDatabase): VectorDatabase object containing vectors to randomly sample distances from.
            n_random (int, optional): number of random vector distances to sample.
            q (float, optional): score quantile (default = 0.99)
            n_trials (int, optional): number of sampling trials to perform (workaround for maxing out `torch.cdist`.)
        """

        # We need a unique maximum distance and quantile score value for each item in the VectorDatabase.
        device = vector_db.vectors.device
        item_max_D = torch.zeros(self.batch[-1] + 1).to(device)
        item_thresholds = torch.zeros_like(item_max_D).to(device)

        for _ in range(n_trials):
            r_idx = torch.randperm(len(vector_db.vectors))[:n_random].to(device)
            random_vecs = vector_db.vectors[r_idx].to(device)

            for item_index in torch.arange(len(item_max_D)):
                item_vectors = self.vectors[self.batch == item_index].to(device)

                # 1 - cos, matching VectorDatabase.search(). Normalised explicitly for the
                # same reason unit_vectors() does it: the cosine must not depend on the
                # encoder's F.normalize still being there.
                iv = torch.nn.functional.normalize(item_vectors, p=2, dim=1)
                rv = torch.nn.functional.normalize(random_vecs, p=2, dim=1)
                D = cos_distance(iv @ rv.t())
                D = D.flatten()
                max_D = D.max()
                S = max_D - D
                threshold = torch.quantile(S, q)

                item_max_D[item_index] += max_D / n_trials
                item_thresholds[item_index] += threshold / n_trials

        self.score_thresholds = (item_max_D, item_thresholds)

    def faiss_index(self, gpu=True):
        """
        Convert self.vectors object into a proper FAISS index for querying.
        Args:
            gpu (bool, optional): indicates use of GPU (default = True).
        Returns:
            faiss index object
        """
        np_vectors = self.unit_vectors()

        dim = np_vectors.shape[1]
        # Inner product on unit-normalised rows IS cosine similarity. NOTE for callers that
        # use this index directly: .search now returns SIMILARITIES (descending, high =
        # near), not squared distances -- pass them through cos_distance() before any
        # arithmetic that assumes 0 is closest. VectorDatabase.search() does this for you.
        index = faiss.IndexFlatIP(dim)

        if gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(np_vectors)
        return index

    def search(self, query_vectors, k=2048):
        """Cosine nearest neighbours, returned on the 0-is-closest convention.

        Returns (D, I) with D = 1 - cos, ascending along each row, matching what
        IndexFlatL2 used to hand back up to a factor of two. Row order is preserved, so
        apply_duplicate_mask's "keep the first occurrence" still keeps the NEAREST hit.
        """
        sim, I = self.faiss_index().search(query_vectors, k)
        return cos_distance(sim), I

    def np_vectors(self):
        """
        return numpy version of self.vectors.
        """
        return self.vectors.detach().cpu().numpy().astype("float32")

    def unit_vectors(self):
        """np_vectors with every row L2-normalised, so an inner product is a cosine.

        The encoders already return unit rows, but relying on that makes the metric here
        depend on a guarantee enforced elsewhere; a variant that skipped F.normalize, or a
        .pyg of pre-normalisation activations, would silently stop being cosine.
        """
        v = self.np_vectors()          # already a fresh float32 copy
        faiss.normalize_L2(v)
        return v

    def get_scores(self, D, I, query_db):
        """
        Provided the results of a FAISS-based nearest neighbors operation, get per-item scores of the queried VectorDatabase.
        Nearest-neighbors operation returns N = 2048 nearest neighbors.

        Args:
            D (np.array): numpy array of distances of 2048 neighbor-distance values per-vector.
            I (np.array): neighbor-vector indices corresponding to values in D.
            query_db (VectorDatabase): VectorDB used as query.
        """
        device = query_db.vectors.device

        S = torch.as_tensor(D).to(device)
        I = torch.as_tensor(I).to(device)

        score_thresholds = [x.to(device) for x in query_db.score_thresholds]
        target_batch = query_db.batch.to(device)

        max_D = score_thresholds[0][target_batch].unsqueeze(1)
        threshold_S = score_thresholds[1][target_batch].unsqueeze(1)

        bias = max_D - threshold_S
        S.neg_().add_(bias).clamp_(min=0)

        # Score value calculated so that the smaller the vector distance, the greater the score.
        # S = torch.clamp(max_D.unsqueeze(1) - D - threshold_S.unsqueeze(1), min=0)

        # get index of VectorDatabase item, rather than index of individual vectors.
        mol_I = self.batch.to(device)[I].long()
        S = apply_duplicate_mask(S, mol_I, batch_size=1024)

        # matches = mol_I.unsqueeze(2) == mol_I.unsqueeze(1)
        # previous_matches = torch.tril(matches, diagonal=-1)
        # duplicate_mask = previous_matches.any(dim=2)

        # S[duplicate_mask] = 0

        target_mol_scores = []

        for t_i in target_batch.unique():
            target_mask = target_batch == t_i
            S_t = S[target_mask].flatten()
            I_t = I[target_mask].flatten()
            mol_I = self.batch.to(device)[I_t].long()
            mol_scores = torch.zeros_like(self.batch.unique()).float().to(device)
            mol_scores.scatter_add_(0, mol_I, S_t)
            target_mol_scores.append(mol_scores)

        target_mol_scores = torch.vstack((target_mol_scores))

        return target_mol_scores

    def query(self, query_db):
        """
        Query `self.vectors` with vectors from another VectorDatabase.

        Args:
            query_db (VectorDatabase): VectorDatabase used to query `self.vectors`.

        Returns:
            (list): List containing for each item in query, a 2-tuple list containing
            tensor of sorted non-zero scores and corresponding item index tensor from queried VectorDatabase.
        """
        D, I = self.search(query_db.unit_vectors(), 2048)

        scores = self.get_scores(D, I, query_db)
        sorted_scores, sorted_index = scores.sort(descending=True, dim=1)

        final_scores = []

        for sc, si in zip(sorted_scores, sorted_index):
            final_scores.append((sc[sc > 0].tolist(), si[sc > 0].tolist()))

        return final_scores

    def nearest_neighbors(self, query_db):
        D, I = self.search(query_db.unit_vectors(), 2048)
        mol_I = self.batch[I].long()

        return D, mol_I
