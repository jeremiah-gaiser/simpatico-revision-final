import argparse
import json
import pickle
import sys
from datetime import datetime
from glob import glob
from os import path
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch_geometric.nn import radius

from simpatico.get_train_set import construct_tv_set
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.utils.utils import get_logger


def add_arguments(parser):
    parser.add_argument(
        "input",
        type=str,
        help="Path to train-eval dataset",
    )
    parser.set_defaults(main=main)
    return parser


def positive_margin_loss(anchors, positives, negatives, m=1.0, d=3):
    positive_distances = torch.norm(anchors - positives, dim=1)
    anchors = anchors.repeat(negatives.size(0) // anchors.size(0), 1)

    negative_distances = torch.norm(anchors - negatives, dim=1)
    positive_loss = torch.clamp(positive_distances - m, min=0)
    negative_loss = torch.clamp(m * d - negative_distances, min=0)

    return positive_loss.mean() + negative_loss.mean()


class HardBatchScheduler:
    def __init__(
        self,
        start_size=32,
        max_size=512,
        growth_factor=2,
        patience=100,
        target_metric=0.15,
    ):
        """
        Args:
            start_size: Initial HARD_BATCH_SIZE (e.g., 32 molecules).
            max_size: Maximum HARD_BATCH_SIZE (e.g., 512 molecules).
            growth_factor: Multiplier for size increase (e.g., 2 -> doubles size).
            patience: Number of batches with good metrics required to level up.
            target_metric: The mean_rank value below which we consider the task 'solved'
                           (0.0 = perfect top-1 rank, 1.0 = worst rank).
        """
        self.current_size = min(start_size, max_size)
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.patience = patience
        self.target_metric = target_metric
        self.win_streak = 0

    def step(self, mean_rank_metric):
        """
        Updates the schedule based on the latest loss metric.
        Returns the new HARD_BATCH_SIZE.
        """
        # Check if model is performing well (Rank is low/good)
        if mean_rank_metric < self.target_metric:
            self.win_streak += 1
        else:
            self.win_streak = 0  # Reset if performance drops

        # Level Up Mechanism
        if self.win_streak >= self.patience:
            old_size = self.current_size
            self.current_size = int(self.current_size * self.growth_factor)

            # Cap at max
            self.current_size = min(self.current_size, self.max_size)

            if self.current_size > old_size:
                print(
                    f"\n[Curriculum] Level Up! Increased Hard Batch Size: {old_size} -> {self.current_size}"
                )
                self.win_streak = 0  # Reset streak for the new difficulty level

        # return self.current_size


def contrastive_loss(
    prot_x,
    prot_pos,
    prot_batch,
    lig_x,
    lig_pos,
    lig_batch,
    hard_l_embeddings,
    prot_anchor=True,
    temperature=0.2,
    temperature_struct=None,
    temperature_hard=None,
    interaction_radius=4.0,
    phys_dist_threshold=6.0,
    pair_index=None,
    window_size=100,
    max_considered_rank=2048,
    use_external=True,
    external_mode="infonce",
    hinge_margin=0.7,
    hinge_weight=1.0,
    window_mode="band",
    exclude_top_frac=0.0,
    exclude_mode="hard",
    lig_sim_clamp=0.0,
    hard_batch=None,
    group_of_complex=None,
    exclude_norm="kept",
    ablate_struct=False,
    struct_weight=1.0,
):
    if pair_index is None:
        p_index, m_index = radius(
            lig_pos,
            prot_pos,
            interaction_radius,
            lig_batch,
            prot_batch,
        )
    else:
        # PLIP-annotated positives. Proximity is a poor proxy for interaction: measured over
        # 3,000 complexes only 17.5% of protein-ligand atom pairs within 4 A belong to any PLIP
        # interaction, so the radius form spends five-sixths of its positive signal pulling
        # together atoms with no reason to be compatible. It also misses 19% of real
        # interactions, though 96% of those are water bridges (median 5.09 A), which are
        # mediated by an intervening water and arguably should not be modelled as contact.
        p_index, m_index = pair_index
    p_embeddings, p_coords, p_batch = (
        prot_x[p_index],
        prot_pos[p_index],
        prot_batch[p_index],
    )
    l_embeddings, l_coords, l_batch = (
        lig_x[m_index],
        lig_pos[m_index],
        lig_batch[m_index],
    )

    if prot_anchor:
        a_embeddings, a_coords, a_batch = p_embeddings, p_coords, p_batch
        s_embeddings, s_coords, s_batch = l_embeddings, l_coords, l_batch
    else:
        a_embeddings, a_coords, a_batch = l_embeddings, l_coords, l_batch
        s_embeddings, s_coords, s_batch = p_embeddings, p_coords, p_batch

    device = prot_x.device
    N = a_embeddings.shape[0]
    M = hard_l_embeddings.shape[0]  # Total Atoms in External Batch

    # --- DEFENSE 1: NaN Checks ---
    if torch.isnan(p_embeddings).any():
        raise ValueError("Critical: NaNs detected in p_embeddings!")

    # --- DEFENSE 2: Force Float32 ---
    with torch.cuda.amp.autocast(enabled=False):
        a_f32 = a_embeddings.float()
        s_f32 = s_embeddings.float()
        hard_l_f32 = hard_l_embeddings.float()

        # Normalize
        # p_f32 = F.normalize(p_f32, p=2, dim=1)
        # l_f32 = F.normalize(l_f32, p=2, dim=1)
        # hard_l_f32 = F.normalize(hard_l_f32, p=2, dim=1)

        # 1. Compute Local Logits (FP32)
        # The two losses take independent temperatures. The raw diagonal similarity is captured
        # BEFORE scaling, because the positive score used by the hard-negative loss must be on
        # the hard temperature's scale -- reusing the struct-scaled diagonal would put positive
        # and negatives on different scales inside one cross-entropy, and whichever temperature
        # was smaller would dominate the loss regardless of the embeddings.
        t_struct = temperature if temperature_struct is None else temperature_struct
        t_hard = temperature if temperature_hard is None else temperature_hard
        logits_local = a_f32 @ s_f32.t()
        pos_raw = logits_local.diagonal().clone()
        logits_local.div_(t_struct)
        ext_raw = a_f32 @ hard_l_f32.t()
        # In-batch pools contain the anchor's OWN complex, so its positive and every near-
        # positive sit in the negative set. `hard_batch` names each pool row's complex; rows
        # from the anchor's complex are removed outright. The ligand pool needs no such mask
        # -- it is drawn from elsewhere and blacklists the anchor's ligand_id at draw time --
        # so hard_batch is None there and this costs nothing.
        if hard_batch is not None:
            with torch.no_grad():
                # `group_of_complex` widens "own complex" to "own binding site": two PDBbind
                # entries of the same pocket are not negatives for each other. None keeps the
                # literal per-complex mask.
                if group_of_complex is None:
                    a_grp, pool_grp = a_batch, hard_batch
                else:
                    a_grp, pool_grp = group_of_complex[a_batch], group_of_complex[hard_batch]
                same_complex = a_grp.unsqueeze(1) == pool_grp.unsqueeze(0)
            ext_raw = ext_raw.masked_fill(same_complex, -1e9)
        logits_external = ext_raw / t_hard

    # --- Masks ---
    # Built in one N*N buffer under no_grad. The masks carry no gradient, and materialising
    # phys_dists / batch_mask / eye_mask separately costs several N*N tensors that peak
    # memory cannot afford at larger batch sizes. Values are identical to the expanded form:
    #   struct_mask = (~batch_mask) | (proximal_mask & ~eye_mask)
    labels_local = torch.arange(N, device=device)
    with torch.no_grad():
        struct_mask = torch.cdist(a_coords, s_coords) < phys_dist_threshold  # proximal_mask
        struct_mask.diagonal().fill_(False)                                  # & ~eye_mask
        struct_mask |= a_batch.unsqueeze(1) != s_batch.unsqueeze(0)          # | ~batch_mask

    # ==============================================================================
    # COMPONENT 1: Structural Loss (Intra-Molecule)
    # ==============================================================================
    # The diagonal is read before the fill, so logits_local can be masked in place rather
    # than copied into a second N*N float tensor.
    pos_logits = (pos_raw / t_hard).unsqueeze(1)
    logits_local.masked_fill_(struct_mask, -1e9)
    loss_struct = F.cross_entropy(logits_local, labels_local)

    # ==============================================================================
    # COMPONENT 2: Balanced Inter-Molecular Loss (Hard Window + Random Buffer)
    # ==============================================================================

    # A. Positive Scores (diagonal of logits_local, captured above before the in-place fill)

    # B. Hard Mining (Rank Window)
    # Search Horizon: limited to 2048 atoms or total size M
    limit_k = min(max_considered_rank, M)
    sorted_logits, _ = torch.topk(logits_external, k=limit_k, dim=1)

    # Calculate Rank relative to Search Horizon
    pos_expanded = pos_logits.expand(-1, limit_k)
    rank_in_topk = (sorted_logits > pos_expanded).sum(dim=1)
    # Calculate Relative Rank as specified
    relative_rank = rank_in_topk.float() / limit_k

    if window_mode == "floor" and window_size > 0:
        # Keep the window's FLOOR, drop its ceiling. The floor is the curriculum: ranks
        # [0, start) are the negatives too hard for THIS anchor given how its own positive is
        # ranking, and skipping them is what the band was really providing. The ceiling at
        # start + window_size was an arbitrary cap on how many of the remaining negatives to
        # score, and at t_hard 0.05 it is unnecessary -- the softmax already concentrates on
        # N_eff ~46, fewer than the band held, so admitting the rest costs nothing and lets
        # the temperature decide the weighting.
        #
        # Removing the window ENTIRELY (window_mode "none", clip_nowindow) was mildly
        # damaging: it also discarded the floor, so every anchor faced the globally hardest
        # negatives from the first epoch. This keeps the easing and drops only the cap.
        #
        # start is computed with the same formula as the band, which guarantees at least
        # window_size negatives survive even for an anchor whose positive is beaten by
        # everything -- a floor of rel * limit_k would mask the whole row at rel = 1.
        start_indices = (relative_rank * (limit_k - window_size)).long()
        below_floor = (torch.arange(limit_k, device=device).unsqueeze(0)
                       < start_indices.unsqueeze(1))
        hard_window_logits = sorted_logits.masked_fill(below_floor, -1e9)
    elif window_size <= 0 or window_mode == "none":
        # No banding: every atom inside the search horizon is a negative. The 100-atom window
        # is explicit hard-negative mining, introduced when t_hard was 0.2 -- there the softmax
        # is flat (N_eff 1,146 of a 4,261-atom pool, top-100 holding only 19.9% of gradient),
        # so the hard negatives have to be picked by hand. At 0.05 the softmax does it: N_eff
        # is 45.7, FEWER than the window holds, and the top-100 by similarity already carry
        # 88.5% of the gradient. Measured in hierarchical/window_vs_temperature.py.
        #
        # Worse, the window is positioned by the anchor's own relative_rank, so a struggling
        # anchor's band slides down the list and misses the hardest negatives -- it captures
        # only 59.9% of the pool's gradient at 0.05. Dropping it hands back that 40%.
        #
        # What is lost is the per-anchor CURRICULUM, which temperature cannot replicate.
        # HardBatchScheduler still provides one at the pool level (32 -> 162 molecules).
        #
        # Cheaper, too: the N x M product is already materialised, so the topk and gather were
        # overhead on a computation that had been done.
        hard_window_logits = sorted_logits
    else:
        # Calculate Linear Window Start Index
        # Maps [0, 1] to [0, limit_k - window_size]
        start_indices = (relative_rank * (limit_k - window_size)).long()

        # Generate the window indices for each row
        offsets = torch.arange(window_size, device=device)
        gather_indices = (start_indices.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            min=0, max=limit_k - 1
        )

        hard_window_logits = torch.gather(sorted_logits, 1, gather_indices)

    # Semantic clamp on the LIGAND-ANCHORED direction only. That term is same-modality --
    # anchor and pool are both ligand atoms -- so every gradient it applies pushes two ligand
    # representations apart, and above a certain cosine the two atoms are the same chemistry.
    # Separating them is the objective dismantling the semantic space it is meant to build.
    # The protein-anchored direction is cross-modal and has no such problem, so it is untouched.
    #
    # Masking to -1e9 removes the term from the softmax DENOMINATOR, which zeroes its gradient
    # exactly (dL/dz_j is its softmax weight). Measured on clip_newgraph_zinc_e25, window
    # negatives above cos 0.90 are 0.04% of terms but 22.5% of the gradient at t_hard 0.05, and
    # 83% of them share the anchor's element; above 0.95 it is 9.9% of gradient at 97.7% same
    # element. The low temperature is what makes this bite -- the softmax concentrates mass
    # exactly where the chemistry signal is. At t_hard 0.2 the same cut spares 0.35%.
    #
    # Threshold is a COSINE: both encoders L2-normalise their output (MolEncoder.py:104,
    # ProteinEncoder.py:192), so these inner products are cosines. sorted_logits are already
    # divided by t_hard, hence the rescale.
    # `hard_batch is None` keeps this to the same-modality case it was measured on. With a
    # protein-atom pool the ligand-anchored term is cross-modal, so a high cosine means a
    # ligand atom and a pocket atom that are compatible, not two copies of the same chemistry,
    # and clamping it would delete the hardest true negatives instead of protecting chemistry.
    clamped_frac = 0.0
    if lig_sim_clamp > 0.0 and not prot_anchor and hard_batch is None:
        too_similar = hard_window_logits > (lig_sim_clamp / t_hard)
        clamped_frac = float(too_similar.float().mean())
        # A row with every negative clamped keeps its positive, so cross_entropy returns ~0
        # for it -- no gradient, no NaN.
        hard_window_logits = hard_window_logits.masked_fill(too_similar, -1e9)

    # D. Random Buffer (The Stabilizer)
    # We sample exactly as many randoms as we have hard window samples to maintain 1:1 balance
    # rand_indices = torch.randint(0, M, (N, hard_window_logits.size(1)), device=device)
    # random_ext_logits = torch.gather(logits_external, 1, rand_indices)

    # Concatenate: [Positive, Hard_Window, Random_Buffer]
    logits_final = torch.cat([pos_logits, hard_window_logits], dim=1)

    # E. Compute Loss
    labels_hard = torch.zeros(N, device=device, dtype=torch.long)

    # Rank-based ROW exclusion. If the positive already ranks inside the top `exclude_top_frac`
    # of the negative pool, this anchor is solved and its row is dropped from the loss
    # entirely -- not just its hardest negatives. The claim InfoNCE makes about such a row is
    # already satisfied, so the remaining gradient only sharpens a margin that is not in
    # dispute, at the cost of diluting the rows that are.
    #
    # relative_rank is the fraction of the pool scoring ABOVE the positive, so
    # `relative_rank < exclude_top_frac` is exactly "the positive is in the top n%".
    #
    # Applied to the external term only: rank is defined against the external pool, and the
    # structural term asks a different question (which pocket atom does this ligand atom
    # belong to) that this rank says nothing about.
    # `exclude_mode` selects how a row's rank becomes its weight. "hard" is the 0/1 cutoff;
    # the rest are continuous in rank and keep "hard" as a limiting case, so a sweep over
    # modes at fixed exclude_top_frac is a sweep over transition sharpness alone.
    #
    # The weighted mean below reduces EXACTLY to the hard form when w is 0/1:
    # (w*ce).sum()/w.sum() is then the mean of ce over the kept rows.
    #
    # Note what no choice of shape can change: 26% of protein-anchored and 33% of
    # ligand-anchored rows sit at rank 0 (the positive beats every pool atom), so any weight
    # that vanishes at rank 0 drops them outright. Softening the cutoff softens the transition
    # band, not the treatment of already-solved rows.
    kept_frac = 1.0
    if exclude_top_frac > 0.0:
        f = exclude_top_frac
        if exclude_mode == "hard":
            w = (relative_rank >= f).float()
        elif exclude_mode == "linear":
            # 0 at rank 0, rising to full weight at the cutoff, flat above it.
            w = (relative_rank / f).clamp(0.0, 1.0)
        elif exclude_mode == "sat":
            # Saturating in RAW rank: k is the rank at which a row earns half weight. Works
            # better than a ramp in relative_rank, which is so skewed (median 0.002-0.005)
            # that a linear ramp spends its whole range inside the first percentile.
            k = max(f * limit_k, 1e-6)
            w = rank_in_topk.float() / (rank_in_topk.float() + k)
        elif exclude_mode == "sigmoid":
            # Smooth step centred on the cutoff, transition width a quarter of it.
            w = torch.sigmoid((relative_rank - f) / (0.25 * f))
        else:
            raise ValueError(f"unknown exclude_mode {exclude_mode!r}")
        kept_frac = float(w.mean())
        ce = F.cross_entropy(logits_final, labels_hard, reduction="none")
        # "kept" renormalises over the surviving rows, so every live row's gradient is scaled
        # by 1/kept_frac -- ~2x when half the rows sit at rank 0, and rising through training as
        # more rows saturate. That is a second factor: a time-varying upweight of the whole
        # external term. "all" divides by N, so excluded rows lose their contribution and every
        # other row's gradient is exactly what it would have been without the exclusion.
        denom = w.sum() if exclude_norm == "kept" else torch.tensor(float(N), device=device)
        # Keeps the graph connected when every weight is zero; a fresh scalar would detach it.
        loss_external = (w * ce).sum() / denom if float(denom) > 0 else logits_final.sum() * 0.0
    else:
        loss_external = F.cross_entropy(logits_final, labels_hard)

    # Metric: Normalized by limit_k (Search Horizon) to prevent "Success Illusion"
    # 0.0 = Best possible rank
    # 1.0 = Positive fell out of the search window (Curriculum Failure)
    mean_rank = rank_in_topk.float().mean().item() / limit_k

    if use_external and external_mode == "hinge":
        # Margin hinge in place of InfoNCE for this direction. InfoNCE asserts that the
        # cross-modal positive must beat EVERY pool atom, which for chemically identical atoms
        # is the wrong claim and measurably scrambles chemistry: with it, protein-anchored
        # mining sits BELOW chance on element match (0.84x pool composition, fluorine anchors
        # drawing zero fluorine negatives). A hinge says only "do not collapse" -- atoms may sit
        # at high similarity up to `hinge_margin` without penalty, and only closer pairs are
        # pushed. Most terms are then zero-gradient, so expect slower convergence than InfoNCE
        # (the rescorer margin variant needed ~3x the schedule to match it).
        loss_external = hinge_weight * F.relu(ext_raw - hinge_margin).mean()
        with torch.no_grad():
            viol = float((ext_raw > hinge_margin).float().mean())
        _ws = 2.0 * struct_weight / (struct_weight + 1.0)
        _we = 2.0 / (struct_weight + 1.0)
        return _ws * loss_struct + _we * loss_external, (
            viol, _ws * loss_struct.item(), _we * loss_external.item(), 1.0, 0.0)

    if not use_external:
        # Ablation: drop the hard-negative term for this direction. With prot_anchor=False the
        # anchor is a LIGAND atom and hard_l_embeddings is also ligand atoms, so that term is
        # the only same-modality one in the loss -- it pushes every ligand contact atom away
        # from every pool ligand atom, including chemically identical ones. Measured on a real
        # training step, the negatives it actually selects ARE the chemically identical ones
        # (fluorine anchors retrieve fluorine at 99x pool base rate), so it works directly
        # against chemical generalisation. The struct term for this direction is kept: it is
        # cross-modal (which pocket atom does this ligand atom belong to).
        # No external term to balance, so struct_weight does not apply here: scaling a lone
        # term is a step-size change, which is what the normalisation exists to avoid.
        return loss_struct, (float("nan"), loss_struct.item(), 0.0, 1.0, 0.0)

    if ablate_struct:
        # Mirror of the `not use_external` branch above: keep the HARD-negative term, drop the
        # SELF term. Self negatives are the anchor's own complex at >= 6 A (struct_mask masks
        # every cross-complex pair), so this leaves the anchor contrasted only against the
        # external pool. loss_struct is still computed -- pos_raw, which the external term's
        # softmax needs, is the diagonal of the same matrix -- it is simply not summed in.
        return loss_external, (mean_rank, 0.0, loss_external.item(), kept_frac, clamped_frac)

    # NORMALISED reweighting. The coefficients sum to 2, exactly as the unweighted
    # `loss_struct + loss_external` does, so struct_weight changes the RATIO between the two
    # terms without changing the total loss scale -- and so without changing the effective step
    # size on the shared encoder parameters. A plain `w * struct + ext` would confound the
    # reweighting with a learning-rate increase. struct_weight 1.0 reproduces the plain sum
    # exactly: both coefficients are 1.
    w_s = 2.0 * struct_weight / (struct_weight + 1.0)
    w_e = 2.0 / (struct_weight + 1.0)
    return w_s * loss_struct + w_e * loss_external, (
        mean_rank,
        w_s * loss_struct.item(),
        w_e * loss_external.item(),
        kept_frac,
        clamped_frac,
    )


def sigreg(z, n_slices=256, t_max=5.0, n_t=17):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun, arXiv 2511.08544).

    Pushes an embedding population toward isotropic N(0, I) by sketching: project onto `n_slices`
    random unit directions and apply a univariate goodness-of-fit test to each 1-D marginal. The
    test is Epps-Pulley, which compares empirical and theoretical characteristic functions,

        EP = N * integral |phi_hat(t) - exp(-t^2/2)|^2 w(t) dt,   w(t) = exp(-t^2/2)

    evaluated by trapezoid over t in [-t_max, t_max]. The leading N is what keeps the value O(1):
    the ECF error is O(1/sqrt(N)), so squaring and multiplying by N removes the batch dependence,
    which matters here because batch size is a variable we change. Gradients are bounded by
    4 sigma^2 / N regardless of the input distribution, so this cannot blow up the way a raw
    covariance penalty can.

    WHY THE INPUT MUST BE PRE-NORMALISATION. Both encoders end in F.normalize, so their outputs
    lie on S^63 and have no chance of being Gaussian -- run directly on them this measures a
    fixed mismatch and its gradient is noise. Feed it the activation BEFORE normalisation, where
    isotropy is exactly the property that makes the shipped rep uniform on the sphere:
    z ~ N(0, s^2 I)  =>  z/||z|| ~ Uniform(S^{K-1}). See `sigreg_space` in main().

    cos/sin rather than the reference implementation's complex exp: identical value, and it keeps
    the graph real so autograd needs no complex support.
    """
    A = torch.randn(z.size(1), n_slices, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_t, device=z.device, dtype=z.dtype)
    phi = torch.exp(-0.5 * t.square())                    # CF of N(0,1); doubles as the weight
    proj = (z @ A).unsqueeze(2) * t                       # [N, M, T]
    ecf_re = proj.cos().mean(0)                           # [M, T]
    ecf_im = proj.sin().mean(0)
    err = ((ecf_re - phi).square() + ecf_im.square()) * phi
    return torch.trapz(err, t, dim=1).mean() * z.size(0)


def margin_loss(
    prot_x,
    prot_pos,
    prot_batch,
    lig_x,
    lig_pos,
    lig_batch,
    pool_embeddings,
    interaction_radius=4.0,
    phys_dist_threshold=6.0,
    margin=0.2,
    self_weight=1.0,
    cross_weight=1.0,
    ext_weight=1.0,
    reduction="active",
    pair_index=None,
):
    """Triplet margin loss, replacing `contrastive_loss` in both anchor directions.

    Why not InfoNCE. The softmax asserts the positive must beat EVERY negative, and the gradient
    it puts on a negative grows with that negative's similarity -- so the more chemically alike a
    negative is, the harder it is pushed away. That is the right objective for retrieval (it
    clears the probe's landing region) and the wrong one for a semantic space, and the space paid
    for it: mined ligand-anchored negatives match the anchor's element BELOW pool base rate
    (0.84x), so the neighbourhood of a positive is not a chemical neighbourhood -- it is whatever
    landed in that bucket. Pair that with a protein voxel whose embedding barely responds to any
    single atom (peak shift 0.0045) but is a sharp function of the surrounding configuration, and
    the model reads as a continuous hash table keyed on pocket geometry.

    The form. One margin, applied as a triplet:

        L = relu(sim(anchor, negative) - sim(anchor, positive) + margin)

    This is the canonical margin loss in metric learning (torch's own TripletMarginLoss, FaceNet,
    Weinberger & Saul), and it is RELATIVE, which is the property that matters here. An absolute
    threshold -- "positives above m, negatives below m" -- imposes a global geometry: every
    contacting pair must reach m, and since one voxel contacts several ligand atoms, that drags
    all of a voxel's partners together whether or not they are alike. Measured, it also does not
    train: on the first attempt the positive similarity pinned at 0.485 against a 0.5 threshold
    and the loss flatlined at 0.04 inside 200 batches, because once a pair is on the correct side
    of an absolute line there is nothing left to learn. The triplet form fixes no absolute scale;
    it only asks that the true partner beat an impostor by `margin`. A chemically similar negative
    may sit arbitrarily close to the positive in absolute terms, which is exactly the
    "negatives that genuinely share characteristics may occupy the proximal region" behaviour the
    absolute form cannot express.

    Reduction. `active` averages over VIOLATING triplets only; `all` averages over every triplet.
    The default is `active` because the pool contributes ~47,000 negatives per anchor, so once
    most are satisfied an all-triplet mean is dominated by zeros and the gradient vanishes --
    Hermans, Beyer & Leibe (2017) make exactly this point against batch-all averaging. With
    `active`, the loss value stops being a progress signal; the violation FRACTION reported
    alongside it is the progress signal.

    No hard-negative mining. Every negative is scored, and the relu keeps only those inside the
    margin -- the hinge selects its own hard negatives. That is also what makes a 2048-molecule
    pool affordable: no topk over the pool, no rank window, and HardBatchScheduler is inert.

    Three negative populations, each normalised and weighted separately so their relative
    contribution does not drift with batch or pool size:
      struct  -- within the batch, the same negative set `contrastive_loss` used: other
                 complexes' atoms, AND atoms of the anchor's own complex more than
                 `phys_dist_threshold` away (the self-negatives).
      ext-P   -- the pool, against a protein anchor. Cross-modal.
      ext-L   -- the pool, against a ligand anchor. The only same-modality term in the loss, and
                 the one InfoNCE was scrambling.

    One call covers both anchor directions: `sim(anchor, positive)` is symmetric under the swap
    and the structural term is its own transpose, so only the external term needs doing twice.
    """
    if pair_index is None:
        p_index, m_index = radius(
            lig_pos, prot_pos, interaction_radius, lig_batch, prot_batch
        )
    else:
        p_index, m_index = pair_index

    p_embeddings, p_coords, p_batch = prot_x[p_index], prot_pos[p_index], prot_batch[p_index]
    l_embeddings, l_coords, l_batch = lig_x[m_index], lig_pos[m_index], lig_batch[m_index]

    if torch.isnan(p_embeddings).any():
        raise ValueError("Critical: NaNs detected in p_embeddings!")

    def _reduce(v, mask):
        """Mean hinge over the triplets `mask` selects, and the fraction of them violating."""
        vm = v * mask
        n_act = int((vm > 0).sum())
        n_val = max(int(mask.sum()), 1)
        denom = n_act if reduction == "active" else n_val
        return vm.sum() / max(denom, 1), n_act / n_val

    with torch.amp.autocast("cuda", enabled=False):
        pe = p_embeddings.float()
        le = l_embeddings.float()
        pool = pool_embeddings.float()

        # The positive similarity of each anchor. No temperature -- a margin on a cosine is
        # already on an interpretable scale, and a temperature would only rescale the margin.
        pos_sim = (pe * le).sum(dim=1)
        # Broadcast against every negative of that anchor. `.detach()` is NOT applied: the
        # gradient that pulls the positive in is the other half of this loss.
        offset = (margin - pos_sim).unsqueeze(1)

        # --- structural negatives, kept as TWO populations --------------------------------
        # self  : same complex, at least phys_dist_threshold away. This is contrastive_loss's
        #         structural negative set exactly. Nearer ligand atoms are EXCLUDED on purpose:
        #         an atom within 6 A of this voxel is a plausible partner for it, and pushing it
        #         away is the over-separation this loss exists to avoid.
        # cross : a different complex's atoms. contrastive_loss leaves these out of its
        #         structural term entirely and handles cross-complex negatives through the pool;
        #         they are kept here but scored and weighted separately, because they are a
        #         different claim (this is the wrong POCKET) from the self negatives (this is
        #         the wrong ATOM in the right pocket) and there are ~8x more of them.
        #
        # This polarity was INVERTED before 2026-09-29: the mask was applied as its own
        # complement, so the loss pushed away the nearby same-complex atoms and never saw the
        # distant ones. Proven behaviourally against contrastive_loss on a crafted pair -- the
        # two returned exactly opposite results on identical input. Every margin-mode run before
        # that date (margin1, margin1_sigreg) trained against the inverted set.
        with torch.no_grad():
            same = p_batch.unsqueeze(1) == l_batch.unsqueeze(0)
            far = torch.cdist(p_coords, l_coords) >= phys_dist_threshold
            self_mask = same & far
            self_mask.diagonal().fill_(False)       # the diagonal is the positive
            cross_mask = ~same
            del same, far

        # relu_ specifically (not clamp_): relu's backward reads its OUTPUT, which an in-place
        # op leaves intact, whereas clamp_ destroys the input it needs. The masks are applied at
        # REDUCTION time rather than to the product, so one hinge buffer serves both populations
        # and nothing writes into a tensor autograd has saved.
        local = pe @ le.t()
        local.add_(offset)
        F.relu_(local)
        loss_self, self_viol = _reduce(local, self_mask)
        loss_cross, cross_viol = _reduce(local, cross_mask)
        del local, self_mask, cross_mask

        # --- pool negatives, both directions ---------------------------------------------
        ext_losses, ext_viols = [], []
        for anchor in (pe, le):
            ext = anchor @ pool.t()
            ext.add_(offset)
            F.relu_(ext)
            l_, v_ = _reduce(ext, torch.ones_like(ext, dtype=torch.bool))
            ext_losses.append(l_)
            ext_viols.append(v_)
            del ext
        loss_ext_p, loss_ext_l = ext_losses

    loss = (self_weight * loss_self + cross_weight * loss_cross
            + ext_weight * (loss_ext_p + loss_ext_l))
    return loss, (
        float(pos_sim.mean().detach()),
        self_viol,
        cross_viol,
        ext_viols[0],
        ext_viols[1],
        loss_self.item(),
        loss_cross.item(),
        loss_ext_p.item(),
        loss_ext_l.item(),
    )


def build_pair_index(protein_batch, molecule_batch, protein_out, device, pocket_radius=6):
    """Concatenated per-graph PLIP pairs, in the coordinates contrastive_loss actually uses.

    Two shifts are needed. Stored indices are per-graph, so they are offset into the batched
    tensors. And the protein row must be remapped from FULL-graph atom indices to VOXEL
    indices: ProteinEncoder emits embeddings only for atoms inside its 6 A pocket mask, and
    that is what contrastive_loss indexes. The original radius() form never needed this
    because it was handed protein_out.pos directly.

    Pairs whose protein atom falls outside the pocket are dropped -- they cannot be scored
    against an embedding that was not produced.

    Pairs are stored as three flat 1-D attributes rather than one `*_index` tensor: PyG
    increments any key containing 'index' by the source graph's node count, which is right
    for the protein row and wrong for the ligand row of a bipartite pair.
    """
    if not hasattr(protein_batch, "plip_p"):
        return None
    n = protein_batch.plip_n.to(device)
    if int(n.sum()) == 0:
        return None

    # Taken from the encoder's own output rather than recomputed here. With surface_only the
    # pocket is no longer a pure radius() result, so re-deriving it would silently disagree.
    keep = getattr(protein_out, "vox_index", None)
    if keep is None:
        keep = radius(molecule_batch.pos, protein_batch.pos, pocket_radius,
                      molecule_batch.batch, protein_batch.batch)[0].unique().sort().values
    if keep.numel() != protein_out.pos.size(0):
        raise RuntimeError("pocket mask no longer matches ProteinEncoder's output")
    row = torch.full((protein_batch.pos.size(0),), -1, dtype=torch.long, device=device)
    row[keep] = torch.arange(keep.numel(), device=device)

    p_off = torch.repeat_interleave(protein_batch.ptr[:-1].to(device), n)
    m_off = torch.repeat_interleave(molecule_batch.ptr[:-1].to(device), n)
    p_idx = row[protein_batch.plip_p.to(device) + p_off]
    m_idx = protein_batch.plip_l.to(device) + m_off
    ok = p_idx >= 0
    if not bool(ok.any()):
        return None
    return p_idx[ok], m_idx[ok]


def training_step(data_loader, protein_encoder, mol_encoder, hard_batch_scheduler,
                  temperature_struct=None, temperature_hard=None, use_plip=False,
                  hard_no_grad=False, max_considered_rank=2048, window_size=100,
                  ablate_lig_external=False, lig_external_mode="infonce",
                  hinge_margin=0.7, hinge_weight=1.0, loss_mode="infonce",
                  window_mode="band",
                  margin=0.2, self_weight=1.0, cross_weight=1.0, ext_weight=1.0,
                  margin_reduction="active", sigreg_weight=0.0, sigreg_slices=256,
                  sigreg_space="pre", exclude_top_frac=0.0, exclude_mode="hard",
                  lig_sim_clamp=0.0, lig_external_pool="ligand", inbatch_mask="complex",
                  exclude_norm="kept", ablate_struct=False, ablate_external=False,
                  struct_weight=1.0):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.25

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)
    random_ligand_batch = data_loader.get_random_ligand_batch(
        hard_batch_scheduler.current_size, molecule_batch.ligand_id
    ).to(device)

    protein_out = protein_encoder(protein_batch, molecule_batch)
    mol_out = mol_encoder(molecule_batch)
    if hard_no_grad:
        # NOT RECOMMENDED -- kept only to reproduce runs/plip_hn2048. Detaching negatives is
        # standard in memory-bank/queue methods (MoCo, InstDisc), where the negatives are STALE
        # and their gradient would be wrong anyway, and where a momentum encoder keeps the
        # queue consistent. Here the pool is re-encoded fresh from the current encoder every
        # step, so the gradient is available and correct; dropping it is a memory shortcut
        # with no compensating mechanism. Measured: it slows learning enough that mean_rank
        # never reaches HardBatchScheduler's 0.10 target, so the pool never left its start
        # size of 32 and the run stalled. Backprop at pool 2048 fits in 33.8 GB. If activations
        # ever do become the binding constraint, use gradient checkpointing or an actual
        # momentum-encoder queue, not a bare detach.
        with torch.no_grad():
            hard_out = mol_encoder(random_ligand_batch)
    else:
        hard_out = mol_encoder(random_ligand_batch)

    # SIGReg on the pocket reps. Computed on protein_out, i.e. one row per VOXEL -- deliberately
    # not on the pair-indexed p_embeddings the losses use, where a voxel appears once per ligand
    # contact and heavily-contacted voxels would dominate the distribution being regularised.
    reg = None
    if sigreg_weight > 0:
        if sigreg_space == "sphere":
            # Test the shipped unit-norm rep directly. For z uniform on S^(K-1), a^T z has
            # variance 1/K and tends to Gaussian as K grows, so sqrt(K)*z has ~N(0,1) marginals
            # exactly when directions are uniform. More targeted than "pre" -- it spends no
            # gradient shaping the norm, which normalisation discards anyway -- at the cost of
            # departing from the paper, whose target is the ambient Gaussian.
            z = protein_out.x * (protein_out.x.size(1) ** 0.5)
        else:
            z = protein_out.x_raw
        reg = sigreg(z.float(), n_slices=sigreg_slices)

    pair_index = build_pair_index(protein_batch, molecule_batch, protein_out, device) if use_plip else None
    if use_plip and pair_index is None:
        return None, None, None

    total_loss = None
    pm_metrics = []

    if loss_mode == "margin":
        # One call, not one per direction: the hinge's positive and structural terms are
        # symmetric under the anchor swap, so running both would only double-weight them.
        total_loss, loss_metrics = margin_loss(
            protein_out.x, protein_out.pos, protein_out.batch,
            mol_out, molecule_batch.pos, molecule_batch.batch,
            hard_out,
            margin=margin, self_weight=self_weight, cross_weight=cross_weight,
            ext_weight=ext_weight, reduction=margin_reduction, pair_index=pair_index,
        )
        if reg is not None:
            total_loss = total_loss + sigreg_weight * reg
            loss_metrics = loss_metrics + (float(reg.detach()),)
        return (
            total_loss,
            (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch),
            [loss_metrics],
        )

    for prot_anchor in [True, False]:
        # Which pool the external term mines. The protein-anchored direction always mines the
        # ligand pool -- cross-modal, which is the screening task. The ligand-anchored
        # direction has only ever had the SAME pool, making it the one same-modality term in
        # the loss. With lig_external_pool="protein_inbatch" it instead mines the pocket atoms
        # of the other complexes already in this batch: cross-modal, the mirror image of what
        # the protein anchors do, and free -- protein_out is computed either way, so this adds
        # one matmul and no forward pass. Those pairs are currently used by nothing: the struct
        # term masks every cross-complex pair out (struct_mask |= a_batch != s_batch).
        if prot_anchor or lig_external_pool == "ligand":
            pool_x, pool_batch, pool_groups = hard_out, None, None
        else:
            pool_x, pool_batch = protein_out.x, protein_out.batch
            pool_groups = (protein_batch.pocket_gid
                           if inbatch_mask == "pocket" else None)

        loss, loss_metrics = contrastive_loss(
            protein_out.x,
            protein_out.pos,
            protein_out.batch,
            mol_out,
            molecule_batch.pos,
            molecule_batch.batch,
            pool_x,
            prot_anchor,
            hard_batch=pool_batch,
            group_of_complex=pool_groups,
            temperature_struct=temperature_struct,
            temperature_hard=temperature_hard,
            # NOT swapped for prot_anchor: p_index/m_index always address protein/ligand
            # respectively, and contrastive_loss swaps anchor vs sample AFTER indexing.
            pair_index=pair_index,
            max_considered_rank=max_considered_rank,
            window_size=window_size,
            use_external=(not ablate_external) and (prot_anchor or not ablate_lig_external),
            ablate_struct=ablate_struct, struct_weight=struct_weight,
            external_mode=("infonce" if prot_anchor else lig_external_mode),
            hinge_margin=hinge_margin, hinge_weight=hinge_weight,
            window_mode=window_mode, exclude_top_frac=exclude_top_frac, exclude_mode=exclude_mode,
            exclude_norm=exclude_norm,
            lig_sim_clamp=lig_sim_clamp,
        )

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

        pm_metrics.append(loss_metrics)

    if reg is not None:
        total_loss = total_loss + sigreg_weight * reg

    return (
        total_loss,
        (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch),
        pm_metrics,
    )


def diag_ranks(D):
    """
    D: [V, V] tensor
    Returns: [V] tensor, where out[v] is the rank position of D[v, v]
             in row v when sorted ascending.
    """
    V = D.shape[0]

    # argsort each row
    sorted_idx = D.argsort(dim=1)  # [V, V]

    # create a mask where sorted_idx[row] == row
    row_idx = torch.arange(V, device=D.device)
    mask = (
        sorted_idx == row_idx[:, None]
    )  # [V, V], True at the diagonal element’s position

    # get the column index where True
    ranks = mask.nonzero(as_tuple=False)[:, 1]  # [V]

    return ranks


def batch_avg_cdist(A, B, A_batch, B_batch, V=None):
    """
    Compute [V, V] matrix where entry (v_a, v_b) is the average
    pairwise distance between all items of A in batch v_a and
    all items of B in batch v_b.
    """
    V = max(A_batch.max(), B_batch.max()).item() + 1

    # pairwise distances
    D = torch.cdist(A, B)  # [M, N]

    # one-hot encodings of batch membership
    A_onehot = F.one_hot(A_batch, V).to(D.dtype)  # [M, V]
    B_onehot = F.one_hot(B_batch, V).to(D.dtype)  # [N, V]

    # total distances per (v_a, v_b)
    # (V,V) = (A^T @ D @ B)
    sums = A_onehot.T @ D @ B_onehot

    # counts per (v_a, v_b)
    counts = A_onehot.sum(0)[:, None] * B_onehot.sum(0)[None, :]

    return sums / counts


class ScreenTest:
    def __init__(self):
        self.prot_embeds = []
        self.prot_batch = []
        self.mol_embeds = []
        self.mol_batch = []

    def add(self, prot_embeds, prot_batch, mol_embeds, mol_batch):
        self.prot_embeds.append(prot_embeds)
        self.mol_embeds.append(mol_embeds)

        if len(self.prot_batch) == 0:
            prot_batch_mod = 0
            mol_batch_mod = 0
        else:
            prot_batch_mod = self.prot_batch[-1][-1] + 1
            mol_batch_mod = self.mol_batch[-1][-1] + 1

        prot_batch += prot_batch_mod
        mol_batch += mol_batch_mod

        self.prot_batch.append(prot_batch)
        self.mol_batch.append(mol_batch)

    def run(self):
        prot_embeds = torch.vstack(self.prot_embeds)
        prot_batch = torch.hstack(self.prot_batch)
        mol_embeds = torch.vstack(self.mol_embeds)
        mol_batch = torch.hstack(self.mol_batch)

        avg_distances = batch_avg_cdist(prot_embeds, mol_embeds, prot_batch, mol_batch)
        rankings = diag_ranks(avg_distances)
        acc = 1 - (rankings.float().mean() / len(rankings))
        return acc.item()


def validate(
    data_loader,
    protein_encoder,
    mol_encoder,
    hard_batch_scheduler,
    difficulty_value=1,
    batch_size=16,
    temperature_struct=None,
    temperature_hard=None,
    use_plip=False,
    hard_no_grad=False,
    max_considered_rank=2048,
    window_size=100,
    ablate_lig_external=False,
    lig_external_mode="infonce",
    hinge_margin=0.7,
    hinge_weight=1.0,
    loss_mode="infonce",
    margin=0.2,
    self_weight=1.0,
    cross_weight=1.0,
    ext_weight=1.0,
    margin_reduction="active",
    lig_external_pool="ligand",
    inbatch_mask="complex",
    ablate_struct=False,
    ablate_external=False,
    struct_weight=1.0,
):
    validation_loss_vals = []
    screen_test = ScreenTest()

    protein_encoder.eval()
    mol_encoder.eval()

    batch_count = max(data_loader.size // batch_size, 1)

    for batch_idx in range(batch_count):
        with torch.no_grad():
            loss, embed_data, _ = training_step(
                data_loader, protein_encoder, mol_encoder, hard_batch_scheduler,
                temperature_struct=temperature_struct, temperature_hard=temperature_hard,
                use_plip=use_plip, hard_no_grad=hard_no_grad,
                max_considered_rank=max_considered_rank, window_size=window_size,
                ablate_lig_external=ablate_lig_external,
                lig_external_mode=lig_external_mode, hinge_margin=hinge_margin,
                hinge_weight=hinge_weight, loss_mode=loss_mode,
                margin=margin, self_weight=self_weight, cross_weight=cross_weight,
                ext_weight=ext_weight, margin_reduction=margin_reduction,
                lig_external_pool=lig_external_pool, inbatch_mask=inbatch_mask,
                ablate_struct=ablate_struct, ablate_external=ablate_external,
                struct_weight=struct_weight
            )
            if loss is None:          # batch held no annotated positives
                continue
            screen_test.add(*embed_data)
            validation_loss_vals.append(loss.item())

    epoch_acc = screen_test.run()
    return sum(validation_loss_vals) / max(1, len(validation_loss_vals)), epoch_acc


def get_tv_sets(data, validation_file, holdout_file=None):
    with open(validation_file) as f_in:
        validation_substrings = [x.strip() for x in f_in.readlines()]

    holdout_substrings = []
    if holdout_file:
        with open(holdout_file) as f_in:
            holdout_substrings = [x.strip() for x in f_in.readlines()]

    validation_index = []
    holdout_index = []

    for hs in validation_substrings:
        for i in range(len(data)):
            if hs in data[i][0].name:
                validation_index.append(i)

    for hs in holdout_substrings:
        for i in range(len(data)):
            if hs in data[i][0].name:
                holdout_index.append(i)

    train_data = []
    validation_data = []

    for idx in range(len(data)):
        if idx in holdout_index:
            continue
        elif idx in validation_index:
            validation_data.append(data[idx])
        else:
            train_data.append(data[idx])

    return train_data, validation_data


def main(args):
    with open(args.input) as json_f:
        train_params = json.load(json_f)

    train_handle = train_params["train_handle"]
    weights_dir = Path(f"{train_params['output_dir']}/weights")
    weights_dir.mkdir(exist_ok=True)
    weight_checkpoint_interval = int(train_params['weight_checkpoint_interval'])

    BATCH_SIZE = train_params["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = f"{train_params['output_dir']}/{train_handle}.o"
    stats_file = f"{train_params['output_dir']}/{train_handle}_stats.pkl"
    log = get_logger(output_file)

    with open(train_params["data_file"], "rb") as train_validate_data:
        data_corpus = pickle.load(train_validate_data)

    train_data, validation_data = get_tv_sets(
        data_corpus, train_params['validation_file'], train_params["holdout_file"]
    )

    # PLIP-annotated positives, replacing the 4 A proximity rule. Attached as three flat 1-D
    # attributes so PyG concatenates them without applying its 'index' auto-increment, which
    # would shift the ligand row by the protein's node count.
    use_plip = bool(train_params.get("plip_pairs_file"))
    if use_plip:
        pp = torch.load(train_params["plip_pairs_file"], weights_only=False)["pairs"]
        def attach(split):
            kept = []
            for prot, lig in split:
                ix = pp.get(prot.name)
                if ix is None or ix.size(1) == 0:
                    continue
                ok = (ix[0] < prot.pos.size(0)) & (ix[1] < lig.pos.size(0))
                if not bool(ok.any()):
                    continue
                ix = ix[:, ok]
                prot.plip_p, prot.plip_l = ix[0].clone(), ix[1].clone()
                prot.plip_n = torch.tensor([ix.size(1)])
                kept.append((prot, lig))
            return kept
        n0, m0 = len(train_data), len(validation_data)
        train_data, validation_data = attach(train_data), attach(validation_data)

    neg_pool = None
    if train_params.get("negative_pool_file"):
        with open(train_params["negative_pool_file"], "rb") as _f:
            neg_pool = pickle.load(_f)
    # Fraction of each negative batch taken from the pool. Defaults to a 50/50 split when a pool
    # is configured; the previous behaviour (pool only) is negative_mix 1.0, which flattened the
    # library -- see ProteinLigandDataLoader.
    neg_mix = float(train_params.get("negative_mix", 0.5 if neg_pool is not None else 0.0))

    surface_only = bool(train_params.get("surface_only", False))
    surf_cache = train_params.get("surface_cache",
                                  str(Path(train_params["data_file"]).with_suffix(".surface.pt")))

    train_loader = ProteinLigandDataLoader(train_data, batch_size=BATCH_SIZE,
                                           negative_pool=neg_pool, negative_mix=neg_mix,
                                           surface=surface_only, surface_cache=surf_cache)
    validation_loader = ProteinLigandDataLoader(validation_data, batch_size=BATCH_SIZE,
                                                negative_pool=neg_pool, negative_mix=neg_mix,
                                                surface=surface_only, surface_cache=surf_cache)

    # Input feature width is config-driven so alternative atom representations can be
    # swapped in (e.g. the 11-class interaction potential plus a coverage flag). Defaults
    # are the original one-hot widths, so existing configs reproduce exactly.
    prot_fd = int(train_params.get("protein_feature_dim", ProteinEncoderDefaults["feature_dim"]))
    mol_fd = int(train_params.get("mol_feature_dim", MolEncoderDefaults["feature_dim"]))
    # Covalent hops kept in the ligand graph (3 = every cached edge, the original behaviour).
    mol_hops = int(train_params.get("mol_max_hops", 3))
    # Pocket cap: at most this many entries from any one pocket per epoch, resampled each
    # epoch. Applied here because it reshapes the loader, but LOGGED in the banner below --
    # the fresh-start branch truncates output_file, so anything logged before it is erased.
    # TRAIN loader only: validation must stay a fixed, full set or per-epoch accuracy stops
    # being comparable across epochs.
    pocket_cap = int(train_params.get("pocket_cap", 0))
    pocket_map = train_params.get("pocket_map",
                                  "/home/exouser/simpatico-ReTrain/pocket_census/pockets_j0.5.json")
    _pc = train_loader.set_pocket_caps(pocket_map, pocket_cap) if pocket_cap else None

    # Embedding width. ONE key drives both encoders: the loss is an inner product between a
    # pocket atom and a ligand atom, so the two sides must agree or nothing is comparable.
    # Changing it changes parameter shapes, so an old checkpoint fails loudly on
    # load_state_dict rather than silently mismatching -- but load_encoders still needs it
    # from arch.json to rebuild.
    out_dim = int(train_params.get("out_dim", ProteinEncoderDefaults["out_dim"]))

    # Pocket architecture. "voxel" is the two-population graph ProteinEncoder has always built;
    # "pocket" is PocketEncoder -- one node population (atoms within pocket_radius of any ligand
    # atom, NO surface filter), one projection, one edge set. surface_only does not apply to it
    # and is rejected rather than ignored, since silently dropping it would mean training a
    # different graph than the config asks for.
    pocket_kind = str(train_params.get("pocket_encoder", "voxel"))
    if pocket_kind == "pocket":
        from simpatico.models.protein_encoder.PocketEncoder import PocketEncoder

        if train_params.get("surface_only"):
            raise ValueError(
                "surface_only is a ProteinEncoder (kind=voxel) setting; PocketEncoder has no "
                "surface filter. Remove the key or set pocket_encoder to 'voxel'."
            )
        edge_r = float(train_params.get("edge_radius", 5.0))
        pocket_r = float(train_params.get("pocket_radius", 6.0))
        protein_encoder = PocketEncoder(feature_dim=prot_fd, out_dim=out_dim,
                                        edge_radius=edge_r, pocket_radius=pocket_r).to(device)
        prot_arch = {"kind": "pocket", "feature_dim": prot_fd, "out_dim": out_dim,
                     "edge_radius": edge_r, "pocket_radius": pocket_r}
    elif pocket_kind == "voxel":
        protein_encoder = ProteinEncoder(feature_dim=prot_fd, surface_only=surface_only,
                                         out_dim=out_dim).to(device)
        prot_arch = {"feature_dim": prot_fd, "surface_only": surface_only, "out_dim": out_dim}
    else:
        raise ValueError(f"unknown pocket_encoder {pocket_kind!r}; expected voxel or pocket")

    mol_encoder = MolEncoder(feature_dim=mol_fd, max_hops=mol_hops, out_dim=out_dim).to(device)

    # Architecture sidecar. surface_only and max_hops change what the graph IS, not what shape
    # the weights are, so a checkpoint trained with them loads without complaint into a default
    # encoder and silently evaluates a different model. Everything downstream should build its
    # encoders through simpatico.models.load_encoders, which reads this file.
    arch = {"protein": prot_arch,
            "molecule": {"feature_dim": mol_fd, "max_hops": mol_hops,
                         "out_dim": out_dim}}
    with open(weights_dir / "arch.json", "w") as _a:
        json.dump(arch, _a, indent=2)

    weights_file_template = str(weights_dir / f"{train_handle}_%s.w")
    train_stats = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "mean_rank": [],
        "hn_batch_size": [],
    }
    epoch_start = 1
    hn_batch_size = 32
    if Path(stats_file).exists():
        with open(stats_file, "rb") as stats_in:
            train_stats = pickle.load(stats_in)

        epoch_start = len(train_stats["train_loss"]) + 1
        protein_model_weights, mol_model_weights = torch.load(
            weights_file_template % "CURRENT"
        )
        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)
        hn_batch_size = train_stats["hn_batch_size"][-1]
    else:
        with open(output_file, "w") as log_out:
            True

    # --- INITIALIZATION ---
    # Start small (32) to let the model learn basic atomic identity.
    hn_max = int(train_params.get("hn_max_size", 162))
    hard_no_grad = bool(train_params.get("hard_no_grad", False))
    # Horizon for hard mining, scaled with the pool. contrastive_loss sorts the top
    # `max_considered_rank` external logits and samples a 100-wide band positioned by the
    # positive's relative rank; holding it at 2048 while the pool grows 12x would shrink the
    # horizon from ~38% of the pool's atoms to ~3%, so the window would sit far deeper into
    # the head and difficulty would rise much faster than the pool size implies. It also
    # normalises mean_rank, so a fixed fraction keeps that metric comparable.
    mcr = int(train_params.get("max_considered_rank", max(2048, round(2048 * hn_max / 162))))
    # Epoch-scheduled ramp, when configured. The metric-driven scheduler is a cliff rather
    # than a dial: it needs `patience` CONSECUTIVE batches under target, so the wait scales
    # ~p^-patience. At patience 10 the pool ran 32 -> 1228 inside one epoch and the model
    # collapsed (probe gap 0.041 vs 0.107 for the capped run, everything pulled into a tighter
    # cone); raising patience instead risks the stall that killed the detached run, where the
    # pool never left 32. A fixed epoch schedule can do neither.
    hn_epoch_step = int(train_params.get("hn_epoch_step", 0))
    hn_start = int(train_params.get("hn_start_size", 32))
    # Growth per scheduled step. The observed regressions coincide with the jumps: the gap
    # fell 0.276 -> 0.258 across the epoch window containing a 2x step to 256. A 1.1 factor
    # makes each increase ~10%, small enough for the self-pacing rank window to absorb
    # between epochs.
    hn_growth = float(train_params.get("hn_growth", 2.0))
    # Mining window. Only `window_size` negatives enter the loss per anchor REGARDLESS of pool
    # size -- growing the pool changes which ones are picked, never how many. At 100 the pool
    # contributes no additional gradient however large it gets. Widening is nearly free: the
    # topk over limit_k is already computed, and logits_final only grows from [N,101] to
    # [N,513].
    win = int(train_params.get("window_size", 100))
    # band  : ranks [start, start+window)      -- the original
    # floor : ranks [start, limit_k)           -- keep the curriculum, drop the cap
    # none  : ranks [0, limit_k)               -- no window at all
    win_mode = str(train_params.get("window_mode", "band"))
    ablate_le = bool(train_params.get("ablate_lig_external", False))
    lig_mode = train_params.get("lig_external_mode", "infonce")
    h_margin = float(train_params.get("hinge_margin", 0.7))
    h_weight = float(train_params.get("hinge_weight", 1.0))
    # "infonce" (default) keeps contrastive_loss. "margin" replaces BOTH directions and both
    # terms with margin_loss's hinge, which needs no mining, so the pool is simply held at
    # negative_pool_size for the whole run.
    loss_mode = train_params.get("loss_mode", "infonce")
    # How far the true partner must beat an impostor, in cosine. Relative, not an absolute
    # threshold -- see margin_loss.
    margin = float(train_params.get("margin", 0.2))
    # "active" averages over violating triplets only. "all" averages over every triplet, which
    # with ~47k pool negatives per anchor is dominated by zeros as soon as most are satisfied.
    margin_red = train_params.get("margin_reduction", "active")
    # Split from the former single struct_weight: the self negatives (wrong atom, right pocket)
    # and cross negatives (wrong pocket) are different claims, there are ~8x more cross cells
    # than self, and their violation fractions are now logged separately so the two can be
    # watched and re-weighted independently. Both default to struct_weight, so a config written
    # against the old key still parses.
    struct_w = float(train_params.get("struct_weight", 1.0))
    self_w = float(train_params.get("self_weight", struct_w))
    cross_w = float(train_params.get("cross_weight", struct_w))
    ext_w = float(train_params.get("ext_weight", 1.0))
    neg_pool_size = int(train_params.get("negative_pool_size", 1024))
    # SIGReg (LeJEPA, arXiv 2511.08544). 0.0 = off, and at 0 no projection is drawn and no term
    # is added, so every existing config reproduces exactly. Measured before adding it: pocket
    # voxel reps carry 85-88% of their variance in 8 of 64 directions (effective rank 8.6 for
    # margin1_e20), so there is real anisotropy to act on -- but across the models measured so
    # far anisotropy is ANTI-correlated with LIT-PCBA EF, so the sign of this intervention is an
    # open question, not an assumed improvement. Treat any run with it on as a test of that.
    # Applied to training only; validate() is left alone so validation loss stays the task loss.
    sigreg_w = float(train_params.get("sigreg_weight", 0.0))
    sigreg_n = int(train_params.get("sigreg_slices", 256))
    sigreg_space = train_params.get("sigreg_space", "pre")
    # Rank-based ROW exclusion: if the positive already ranks inside the top `exclude_top_frac`
    # of the pool, drop that anchor's row from the external loss. 0.0 = previous behaviour.
    excl_top = float(train_params.get("exclude_top_frac", 0.0))
    excl_mode = str(train_params.get("exclude_mode", "hard"))
    # Denominator for the excluded loss: "kept" (the original, mean over surviving rows) or
    # "all" (divide by N, so exclusion removes rows without rescaling the rest). See the
    # comment at the denominator in contrastive_loss.
    # Loss-term ablation, BOTH anchor directions. "self" negatives are the struct term (the
    # anchor's own complex at >= 6 A); "hard" negatives are the external term (the ligand pool
    # for protein anchors, in-batch pocket atoms for ligand anchors).
    abl_struct = bool(train_params.get("ablate_struct", False))
    abl_ext = bool(train_params.get("ablate_external", False))
    if abl_struct and abl_ext:
        raise ValueError("ablate_struct and ablate_external together leave no loss")
    excl_norm = str(train_params.get("exclude_norm", "kept"))
    if excl_norm not in ("kept", "all"):
        raise ValueError(f"unknown exclude_norm {excl_norm!r}")
    # Cosine above which a ligand-anchored negative is treated as the same chemistry and
    # dropped from the softmax denominator. 0.0 = off, reproducing prior behaviour exactly.
    lig_clamp = float(train_params.get("lig_sim_clamp", 0.0))

    # Which pool the LIGAND-ANCHORED external term mines.
    #   "ligand"           -- the random-ligand pool, same as the protein-anchored direction.
    #                         Same-modality: anchor and negatives are both ligand atoms.
    #   "protein_inbatch"  -- the pocket atoms of the OTHER complexes in the same batch, with
    #                         the anchor's own complex masked out. Cross-modal, and the mirror
    #                         of what the protein anchors already do. Costs one matmul: those
    #                         embeddings are computed for the struct term regardless.
    lig_pool = str(train_params.get("lig_external_pool", "ligand"))
    if lig_pool not in ("ligand", "protein_inbatch"):
        raise ValueError(f"unknown lig_external_pool {lig_pool!r}")
    # What counts as "not a negative" inside that pool. "complex" masks only the anchor's own
    # entry; "pocket" masks every entry of the same binding site, which at batch 64 reaches
    # ~10% of anchors (3.4 colliding pairs per batch) -- see set_pocket_ids.
    inbatch_mask = str(train_params.get("inbatch_mask", "complex"))
    if inbatch_mask not in ("complex", "pocket"):
        raise ValueError(f"unknown inbatch_mask {inbatch_mask!r}")
    if inbatch_mask == "pocket" and lig_pool == "protein_inbatch":
        _pid = train_loader.set_pocket_ids(pocket_map)
        validation_loader.set_pocket_ids(pocket_map)
        log.info(f"in-batch negatives masked by POCKET: {_pid['pockets']:,} sites, "
                 f"{_pid['unmapped']:,} unmapped entries given singleton ids")
    if lig_pool == "protein_inbatch" and ablate_le:
        raise ValueError("ablate_lig_external removes the term lig_external_pool configures")
    if lig_pool == "protein_inbatch" and lig_clamp > 0.0:
        raise ValueError(
            "lig_sim_clamp guards a SAME-modality term; with a protein pool the "
            "ligand-anchored term is cross-modal and the clamp would cut true negatives"
        )
    if loss_mode == "margin":
        hn_batch_size, hn_max, hn_epoch_step = neg_pool_size, neg_pool_size, 0
    batch_scheduler = HardBatchScheduler(
        start_size=hn_batch_size,
        max_size=hn_max,
        growth_factor=1.5,  # Increase by 50% each time
        patience=10,  # Require 50 stable batches before increasing
        target_metric=0.10,  # Target: Positive is in the top 10% of candidates
    )

    # independent temperatures for the two contrastive losses; both default to the previous
    # single value so existing configs reproduce exactly
    t_struct_cfg = train_params.get("temperature_struct", train_params.get("temperature", 0.2))
    t_hard_cfg = train_params.get("temperature_hard", train_params.get("temperature", 0.2))
    # Logged here, not at construction: the fresh-start branch above truncates output_file
    # with open(..., "w"), which would erase anything written earlier.
    log.info(f"feature dims: protein {prot_fd} | ligand {mol_fd} | embedding {out_dim}-d")
    if use_plip:
        npos = sum(int(g.plip_n) for g in train_loader.proteins)
        log.info(f"positives: PLIP-annotated ({train_params['plip_pairs_file']}) | "
                 f"train {len(train_data)}/{n0} complexes kept, validation "
                 f"{len(validation_data)}/{m0}, {npos:,} positive pairs "
                 f"({npos/max(1,len(train_data)):.1f}/complex)")
    else:
        log.info("positives: 4 A proximity (default)")
    log.info("negatives drawn from: " + (
        f"{neg_mix:.0%} {train_params['negative_pool_file']} ({len(neg_pool):,} molecules) / "
        f"{1-neg_mix:.0%} PDBbind crystal ligands" if neg_pool is not None
        else "PDBbind training ligands (default)"))
    if struct_w != 1.0:
        log.info(f"struct:external ratio {struct_w}:1, NORMALISED "
                 f"(struct x{2*struct_w/(struct_w+1):.4f}, external x{2/(struct_w+1):.4f}; "
                 f"coefficients sum to 2, as at weight 1.0)")
    if abl_struct or abl_ext:
        log.info("LOSS ABLATION: " + ("self/struct term DROPPED, hard-negative term only"
                 if abl_struct else "hard-negative term DROPPED, self/struct term only")
                 + " -- both anchor directions")
    if lig_pool == "protein_inbatch":
        log.info("ligand-anchored external term mines IN-BATCH POCKET ATOMS "
                 "(cross-modal, own complex masked out), not the ligand pool")
    if pocket_kind == "pocket":
        log.info(f"ligand graph: {mol_hops}-hop edges | pocket graph: SINGLE node set, atoms "
                 f"within {pocket_r} A of any ligand atom (no surface filter), one edge set "
                 f"at {edge_r} A")
    else:
        log.info(f"ligand graph: {mol_hops}-hop edges | "
                 f"protein graph: {'SURFACE atoms only' if surface_only else 'all atoms'}")
    if loss_mode == "margin":
        log.info(f"loss: TRIPLET margin {margin} (positive must beat negative by this "
                 f"much) | reduction over {margin_red} triplets | weights self "
                 f"{self_w} / cross {cross_w} / external {ext_w} | pool {neg_pool_size} "
                 f"fixed, no mining")
    log.info(f"hard negatives: max pool {hn_max}, horizon {mcr}, window {win}, "
             f"growth {hn_growth}/{hn_epoch_step or '-'}ep, "
             f"{'detached' if hard_no_grad else 'backprop'}"
             + (" | ABLATED: ligand-anchored external term" if ablate_le else "")
             + (f" | ligand-anchored external: HINGE m={h_margin} w={h_weight}"
                if lig_mode == "hinge" else ""))
    log.info(f"temperature: struct {t_struct_cfg} | hard-negative {t_hard_cfg}")
    if win_mode == "floor" and win > 0:
        log.info(f"hard window FLOOR-ONLY: negatives harder than each anchor's own floor are "
                 f"masked, everything below it admitted (up to {mcr}); weighting left to the "
                 f"softmax at t_hard {t_hard_cfg}")
    elif win <= 0 or win_mode == "none":
        log.info(f"hard window DISABLED: every atom within the {mcr}-atom horizon is a "
                 f"negative, selection left to the softmax at t_hard {t_hard_cfg}")
    if lig_clamp:
        log.info(f"ligand-anchored semantic clamp: negatives above cos {lig_clamp} are dropped "
                 f"from the softmax denominator (same-chemistry pairs)")
    if excl_top:
        log.info(f"rank exclusion: mode {excl_mode}, cutoff {excl_top:.2%} of the pool "
                 f"({'rows dropped outright' if excl_mode == 'hard' else 'rows weighted by rank'}"
                 f"; normalised over {'KEPT rows' if excl_norm == 'kept' else 'ALL N rows'})")
    if sigreg_w:
        log.info(f"SIGReg: weight {sigreg_w} | {sigreg_n} slices | space {sigreg_space} ("
                 + ("pre-normalisation activation" if sigreg_space == "pre"
                    else "unit-sphere rep x sqrt(K)") + ")")

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=train_params["learning_rate"],
    )

    # Linear LR warmup, counted in optimizer steps. Default 0 = off, so every existing config
    # reproduces exactly; train.py otherwise holds LR constant for the whole run.
    # It exists for the linearly-scaled-LR case: raising batch_size and learning_rate by the
    # same factor is the standard way to keep total parameter movement fixed when a batch grows
    # (Goyal et al. 2017), but the scaled LR is only safe once the gradient estimate is stable,
    # and the opening steps are exactly when it is not -- that recipe is specified WITH warmup,
    # not as an alternative to it.
    warmup_steps = int(train_params.get("lr_warmup_steps", 0))
    base_lr = float(train_params["learning_rate"])
    # Resuming mid-run must not replay the ramp, so seed the counter from the epochs already done.
    global_step = (epoch_start - 1) * (train_loader.size // BATCH_SIZE)

    # Exponential moving average of the weights (Polyak averaging). 0.0 = off, and at 0 no
    # shadow is allocated and no extra checkpoint is written, so existing configs are unchanged.
    #
    # Express the decay as a horizon in OPTIMIZER STEPS, not epochs: steps/epoch is
    # size//batch_size, so a fixed decay means a 4x shorter memory at batch 64 than at batch 16
    # and the two runs would not be averaging comparable trajectories. ema_horizon is the
    # ~1/(1-d) window; ema_decay overrides it directly if given.
    ema_horizon = int(train_params.get("ema_horizon", 0))
    ema_decay = float(train_params.get("ema_decay", 1 - 1 / ema_horizon if ema_horizon else 0.0))
    ema_shadow = None
    if ema_decay:
        ema_shadow = [{k: v.detach().clone().float() for k, v in m.state_dict().items()}
                      for m in (protein_encoder, mol_encoder)]
        ema_ckpt = Path(weights_file_template % "CURRENT_ema")
        if epoch_start > 1 and ema_ckpt.exists():
            for sh, sd in zip(ema_shadow, torch.load(ema_ckpt, map_location=device)):
                for k in sh:
                    sh[k].copy_(sd[k])
        log.info(f"weight EMA: decay {ema_decay:.6f} "
                 f"(~{1/(1-ema_decay):.0f}-step window, "
                 f"{1/(1-ema_decay)/max(train_loader.size // BATCH_SIZE,1):.1f} epochs)")
    if _pc:
        log.info(f"pocket cap {pocket_cap}/epoch: {_pc['pockets']:,} pockets "
                 f"(largest {_pc['largest_pocket']}, {_pc['unmapped']:,} unmapped) | "
                 f"epoch {_pc['epoch_size']:,} of {_pc['full_size']:,} entries "
                 f"({100*_pc['epoch_size']/_pc['full_size']:.1f}%), resampled each epoch")
    log.info(f"optimizer: AdamW lr {base_lr} | batch {BATCH_SIZE} "
             f"({train_loader.epoch_size // BATCH_SIZE} steps/epoch) | "
             + (f"linear warmup over {warmup_steps} steps" if warmup_steps else "no warmup"))

    for epoch in range(epoch_start, train_params["epochs"] + 1):
        if hn_epoch_step:
            batch_scheduler.current_size = min(
                hn_max, int(round(hn_start * hn_growth ** ((epoch - 1) // hn_epoch_step))))
            # NB: setting win_streak negative does NOT disable step() -- it resets the streak
            # to 0 on any batch above target, wiping the sentinel, after which it levels up
            # normally. step() is skipped outright below instead.
        log.info(f"Epoch {epoch}" + (f" | hard pool {batch_scheduler.current_size}"
                                     if hn_epoch_step else ""))

        epoch_loss_vals = []
        epoch_rank_vals = []

        batch_loss_vals = []
        batch_rank_vals = []

        protein_encoder.train()
        mol_encoder.train()

        for batch_idx in range(train_loader.epoch_size // BATCH_SIZE):
            loss, _, loss_metrics = training_step(
                train_loader, protein_encoder, mol_encoder, batch_scheduler,
                temperature_struct=t_struct_cfg, temperature_hard=t_hard_cfg,
                use_plip=use_plip, hard_no_grad=hard_no_grad, max_considered_rank=mcr,
                window_size=win, ablate_lig_external=ablate_le,
                lig_external_mode=lig_mode, hinge_margin=h_margin, hinge_weight=h_weight,
                loss_mode=loss_mode, margin=margin,
                self_weight=self_w, cross_weight=cross_w,
                ext_weight=ext_w, margin_reduction=margin_red,
                sigreg_weight=sigreg_w, sigreg_slices=sigreg_n, sigreg_space=sigreg_space,
                window_mode=win_mode, exclude_top_frac=excl_top, exclude_mode=excl_mode,
                exclude_norm=excl_norm,
                ablate_struct=abl_struct, ablate_external=abl_ext,
                struct_weight=struct_w,
                lig_sim_clamp=lig_clamp, lig_external_pool=lig_pool,
                inbatch_mask=inbatch_mask
            )
            if loss is None:
                continue

            mean_rank = loss_metrics[0][0]
            if not hn_epoch_step and loss_mode != "margin":
                batch_scheduler.step(mean_rank)

            batch_loss_vals.append(loss)
            batch_rank_vals.append(mean_rank)

            if batch_idx % 100 == 0:
                batch_loss_avg = torch.hstack(batch_loss_vals).mean().item()
                batch_rank_avg = torch.tensor(batch_rank_vals).mean().item()
                if loss_mode == "margin":
                    # The rank metrics mean nothing without mining. What matters is whether the
                    # margins are being met: mean positive cosine, and the fraction of negatives
                    # still inside the margin -- structural, external vs a protein anchor, and
                    # external vs a ligand anchor.
                    # With reduction="active" the loss value is not a progress signal --
                    # it is the mean depth of the surviving violations. The violation
                    # FRACTIONS are the progress signal.
                    m = loss_metrics[0]
                    log.info(f"Epoch {epoch}, batch {batch_idx} loss: "
                             f"{round(batch_loss_avg, 4)} | pos {m[0]:+.3f} | viol self "
                             f"{m[1]:.4f} cross {m[2]:.4f} ext-P {m[3]:.4f} ext-L {m[4]:.4f}"
                             f" | L {m[5]:.3f}/{m[6]:.3f}/{m[7]:.3f}/{m[8]:.3f}"
                             + (f" | sigreg {m[9]:.4f}" if len(m) > 9 else ""))
                else:
                    log_string = (
                        "Epoch %s, batch %s loss: %s, Mean rank: %s, HN batch size: %s"
                        + (", rows kept: %.3f/%.3f"
                           % (loss_metrics[0][3], loss_metrics[1][3])
                           if excl_top and len(loss_metrics[0]) > 3 else "")
                    )
                    log.info(
                        log_string
                        % (
                            epoch,
                            batch_idx,
                            round(batch_loss_avg, 3),
                            round(batch_rank_avg, 3),
                            batch_scheduler.current_size,
                        )
                    )

                batch_loss_vals = []
                batch_rank_vals = []

                epoch_loss_vals.append(batch_loss_avg)
                epoch_rank_vals.append(batch_rank_avg)

            loss.backward()
            global_step += 1
            if warmup_steps and global_step <= warmup_steps:
                for _g in optimizer.param_groups:
                    _g["lr"] = base_lr * global_step / warmup_steps
            optimizer.step()
            if ema_shadow is not None:
                # Warm the decay in over the first steps, otherwise the shadow stays anchored to
                # the random initialisation for its whole window and is useless for that long.
                d = min(ema_decay, (1 + global_step) / (10 + global_step))
                with torch.no_grad():
                    for sh, mod in zip(ema_shadow, (protein_encoder, mol_encoder)):
                        for k, v in mod.state_dict().items():
                            if sh[k].is_floating_point():
                                sh[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                            else:
                                sh[k].copy_(v)          # counters/buffers: track, do not average
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        epoch_train_loss = torch.tensor(epoch_loss_vals).mean().item()
        epoch_rank = torch.tensor(epoch_rank_vals).mean().item()
        epoch_loss_vals = []
        epoch_rank_vals = []

        epoch_validation_loss, epoch_acc = validate(
            validation_loader, protein_encoder, mol_encoder, batch_scheduler,
            temperature_struct=t_struct_cfg, temperature_hard=t_hard_cfg,
            use_plip=use_plip, hard_no_grad=hard_no_grad, max_considered_rank=mcr,
            window_size=win, ablate_lig_external=ablate_le,
            lig_external_mode=lig_mode, hinge_margin=h_margin, hinge_weight=h_weight,
            loss_mode=loss_mode, margin=margin, self_weight=self_w, cross_weight=cross_w,
            ext_weight=ext_w, margin_reduction=margin_red,
            lig_external_pool=lig_pool, inbatch_mask=inbatch_mask,
            ablate_struct=abl_struct, ablate_external=abl_ext,
            struct_weight=struct_w
        )

        log.info(
            f"Epoch {epoch} validation loss: {epoch_validation_loss}, accuracy: {epoch_acc}"
        )
        for k, v in zip(
            [
                "train_loss",
                "validation_loss",
                "validation_accuracy",
                "mean_rank",
                "hn_batch_size",
            ],
            [
                epoch_train_loss,
                epoch_validation_loss,
                epoch_acc,
                epoch_rank,
                batch_scheduler.current_size,
            ],
        ):
            train_stats[k].append(v)

        with open(stats_file, "wb") as stats_out:
            pickle.dump(train_stats, stats_out)

        torch.save(
            [protein_encoder.state_dict(), mol_encoder.state_dict()],
            weights_file_template % "CURRENT",
        )

        if ema_shadow is not None:
            torch.save(ema_shadow, weights_file_template % "CURRENT_ema")

        if epoch % weight_checkpoint_interval == 0:
            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                weights_file_template % f"e{epoch}",
            )
            # Written as a SEPARATE checkpoint rather than replacing the raw one, so every run
            # yields both curves and the averaging can be evaluated rather than assumed.
            # arch.json sits beside them and applies to both, so load_encoders works unchanged.
            if ema_shadow is not None:
                torch.save(ema_shadow, weights_file_template % f"e{epoch}_ema")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Context already set, which is fine
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
