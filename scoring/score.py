"""Scoring paths compared on one checkpoint: what the retrieval protocol does to the ranking.

Everything reported in this project is EXACT scoring -- every molecule in the library ranked by
    score(m) = mean over pocket voxels of max over atoms of cos(voxel, atom)
The deployed protocol differs in TWO ways at once, and the middle column separates them:

  exact_full       every molecule, mean-of-max-cos        <- what every reported number is
  exact_retrieved  the retrieved subset, mean-of-max-cos  <- candidate set only
  vdb              the retrieved subset, deployed score   <- candidate set + ranking statistic
  hungarian        the retrieved subset, rescored by optimal one-to-one assignment of ligand
                   atoms to pocket voxels; score is the MEAN cosine over matched pairs
  ceiling          not a ranking: the EF a PERFECT rescorer of the retrieved set could reach,
                   i.e. every retrieved active inside the top-k cut. Bounds the rest.

CANDIDATE SET.  VectorDatabase keeps the top-2048 FAISS neighbours per voxel and clamps
anything below a background quantile to zero, so a molecule clearing nothing is never ranked.
RANKING STATISTIC.  The deployed score is sum over voxels of clamp(max_D - distance - thr), a
sum of threshold-clipped margins -- not a mean of max-cosines. It rewards touching many voxels.
ASSIGNMENT.  Both max-based scores let one good voxel be the argmax for many atoms and one good
atom win many voxels, so a single well-placed fragment is counted repeatedly.
linear_sum_assignment forbids that. The mean over matched pairs is the ligand-anchored
denominator -- every atom matched exactly once when n_atoms <= n_vox -- which ef_anchor.py
measured to carry no size bias, unlike the pocket-anchored mean (Spearman +0.158 with size).
Ligands with more heavy atoms than the pocket has voxels leave atoms unmatched and unpaid for;
scipy matches min(n_atoms, n_vox). Left as is deliberately, and counted.

EF is computed against the FULL library size everywhere, counting un-retrieved molecules as
ranked below every retrieved one -- what the deployed pipeline effectively does -- so every
column is directly comparable to exact_full.

FAISS index geometry: this repo ships the COSINE index. The published numbers were generated
with the Euclidean one, and the two rank identically -- on unit vectors d^2 = 2(1-cos), and
max_D, the distance and the threshold all scale by the same 2, so every molecule's score scales
by exactly 2 and the ordering is unchanged. Verified: DEKOIS CDK2 s90 seed 42 gives EF1% 27.50
under both. The cosine form is kept because it is the geometry the model was trained in.

usage: dual_path.py <weights.w> <out.json> [--targets ...] [--quantile 0.99]
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{R}/Simpatico")
from torch_geometric.data import Batch, Data
from simpatico.convert import get_xyz_from_file
from simpatico.decoy_eval import evaluate_mols, get_graph_dict, process_input_file
from simpatico.models import load_encoders
from simpatico.utils.faiss_utils import VectorDatabase

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("out")
ap.add_argument("--targets", nargs="*", default=None)
ap.add_argument("--quantile", type=float, default=0.99)
ap.add_argument("--screen", default="/home/jgaiser/decoy_datasets/litpcba/screen.csv")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = torch.device(a.device)


def score_exact(px, X, batch, n_mol):
    """mean over voxels of the best cosine to any atom -- litpcba_ef_exact.py's score()."""
    bd = torch.searchsorted(batch, torch.arange(n_mol + 1, device=dev))
    out = torch.empty(n_mol, device=dev)
    for s in range(0, n_mol, a.chunk):
        e = min(s + a.chunk, n_mol)
        i, j = int(bd[s]), int(bd[e])
        sim = (px @ X[i:j].t()).t()
        acc = torch.full((e - s, px.size(0)), -1e4, device=dev)
        out[s:e] = acc.index_reduce_(0, batch[i:j] - s, sim, "amax",
                                     include_self=True).mean(1)
    return out


def metrics(scores, is_act):
    """EF/AUC from a score vector over the WHOLE library, higher is better.

    A molecule the retrieval never returned is given -inf, which ties it with every other
    un-retrieved molecule. The tie matters: ranking un-retrieved ACTIVES below un-retrieved
    inactives would be a choice nothing in the protocol makes, and it drags AUC down by a
    third. AUC therefore uses average ranks over ties (the Mann-Whitney definition).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores); na = int(is_act.sum()); nd = n - na
    rk = rankdata(scores, method="average")              # 1 = worst, n = best
    auc = float((rk[is_act].sum() - na * (na + 1) / 2) / (na * nd))
    out = {"auc": auc, "n_act": na, "n_total": n}
    order = np.argsort(-scores, kind="stable")
    finite = np.isfinite(scores)
    n_ret = int(finite.sum())
    for p_ in (0.001, 0.01, 0.05):
        k = max(1, int(round(p_ * n)))
        if k <= n_ret:
            hit = float(is_act[order[:k]].sum())
            exact_cut = True
        else:
            # The cut runs past everything retrieval returned, into the tied block of molecules
            # it never scored. Those are genuinely unordered, so count the retrieved actives
            # plus the EXPECTED actives in the remaining slots drawn at random from the tie --
            # the convention the average-rank AUC above already uses. Flagged, not hidden.
            tie = ~finite
            hit = float(is_act[finite].sum())
            hit += (k - n_ret) * (is_act[tie].sum() / max(1, tie.sum()))
            exact_cut = False
        out[f"ef{p_}"] = float(hit / na / p_)
        out[f"hit{p_}"] = hit
        out[f"ef{p_}_exact_cut"] = exact_cut
    out["n_ranked"] = n_ret
    return out


pe, me = load_encoders(a.weights, device=dev)
pe.eval(); me.eval()
print(f"{a.weights.split('/')[-1]} | quantile {a.quantile}", flush=True)

# decoy_eval derives a target name as row[0].split("/")[-2]. For LIT-PCBA and DUD-E that is
# the target directory and is correct; for DEKOIS it is the literal directory "targets", so all
# 81 collapse to one key. Recover DEKOIS names from column 3 (.../actives/CDK2.sdf -> CDK2).
_rows = process_input_file(a.screen)
_names = [r[0].split("/")[-2] for r in _rows]
if len(set(_names)) < len(_names):
    _alt = [r[2].strip().split("/")[-1].rsplit(".", 1)[0] for r in _rows]
    assert len(set(_alt)) == len(_alt), "target names collapse on both columns"
    print(f"target names recovered from the actives column ({len(_alt)} targets)", flush=True)
    _names = _alt

res = []
for row, t in zip(_rows, _names):
    if a.targets and t not in a.targets:
        continue
    t0 = time.time()
    gd = get_graph_dict(row); tg, _, ag, dg = [gd[x] for x in row]
    tg = tg.to(dev)
    n_act = int(ag.batch[-1]) + 1
    n_dec = int(dg.batch[-1]) + 1
    n_tot = n_act + n_dec

    with torch.no_grad():
        pc = Data(pos=get_xyz_from_file(row[1]))
        pc.batch = torch.zeros(len(pc.pos), dtype=torch.long)
        px = pe(tg, pc.to(dev)).x.float()
        ae = evaluate_mols(ag, me, dev); ae.source = ag.source
        de = evaluate_mols(dg, me, dev); de.source = dg.source

        # ---- path 1: exact over the whole library
        s_act = score_exact(px, ae.x.float().to(dev), ag.batch.to(dev), n_act)
        s_dec = score_exact(px, de.x.float().to(dev), dg.batch.to(dev), n_dec)
        s_all = torch.cat([s_act, s_dec]).cpu().numpy()

        # Atom embeddings in one block with a molecule index per atom, kept for the assignment
        # pass. Decoy molecule indices are offset to continue the actives' numbering, which is
        # the order s_all and is_act use.
        X_all = torch.cat([ae.x.float(), de.x.float()]).to(dev)
        b_all = torch.cat([ag.batch.to(dev), dg.batch.to(dev) + n_act])
        bounds = torch.searchsorted(b_all, torch.arange(n_tot + 1, device=dev)).cpu().numpy()
        enc = time.time() - t0

        # ---- path 2: the deployed retrieval
        tb = Batch.from_data_list([Data(x=px, source=tg.source)])
        allm = Batch.from_data_list(ae.to_data_list() + de.to_data_list())
        tdb = VectorDatabase(tb); vdb = VectorDatabase(allm)
        tdb.get_score_thresholds(vdb, q=a.quantile)
        vscore, vidx = vdb.query(tdb)[0]
        del ae, de, allm, tb, tdb, vdb; torch.cuda.empty_cache()
        t_ret = time.time() - t0 - enc

        # ---- path 4: optimal assignment, retrieved molecules only
        th = time.time()
        vi = np.asarray(vidx, dtype=np.int64)
        hun = np.full(n_tot, -np.inf)
        n_truncated = 0
        for c0 in range(0, len(vi), a.chunk):
            blk = vi[c0:c0 + a.chunk]
            rows = np.concatenate([np.arange(bounds[m], bounds[m + 1]) for m in blk])
            sims = (px @ X_all[torch.as_tensor(rows, device=dev)].t()).t().cpu().numpy()
            off = 0
            for m in blk:
                k_at = int(bounds[m + 1] - bounds[m])
                sub = sims[off:off + k_at]                 # (atoms, voxels)
                off += k_at
                if k_at == 0:
                    continue
                r_i, c_i = linear_sum_assignment(sub, maximize=True)
                hun[m] = float(sub[r_i, c_i].mean())
                if k_at > sub.shape[1]:
                    n_truncated += 1
            del sims
        t_hun = time.time() - th
        del X_all; torch.cuda.empty_cache()

    is_act = np.zeros(n_tot, bool); is_act[:n_act] = True
    n_ret = len(vi)
    act_ret = int(is_act[vi].sum())

    sc_sub = np.full(n_tot, -np.inf); sc_sub[vi] = s_all[vi]
    sc_vdb = np.full(n_tot, -np.inf); sc_vdb[vi] = np.asarray(vscore, dtype=float)

    m_full = metrics(s_all, is_act)
    m_sub = metrics(sc_sub, is_act)
    m_vdb = metrics(sc_vdb, is_act)
    m_hun = metrics(hun, is_act)
    ceil = {}
    for p_ in (0.001, 0.01, 0.05):
        kk = max(1, int(round(p_ * n_tot)))
        ceil[f"ef{p_}"] = float(min(act_ret, kk) / n_act / p_)

    rho = float(spearmanr(s_all[vi], np.asarray(vscore, dtype=float)).statistic) \
        if n_ret > 2 else float("nan")
    rho_h = float(spearmanr(s_all[vi], hun[vi]).statistic) if n_ret > 2 else float("nan")
    k = min(100, n_ret)
    sub_order = vi[np.argsort(-s_all[vi])]
    overlap = len(set(sub_order[:k].tolist()) & set(vi[:k].tolist())) / k if k else float("nan")

    r = {"target": t, "n_total": n_tot, "n_act": n_act, "n_retrieved": n_ret,
         "frac_retrieved": n_ret / n_tot, "act_retrieved": act_ret,
         "frac_act_retrieved": act_ret / n_act,
         "retrieval_enrichment": (act_ret / n_act) / (n_ret / n_tot),
         "spearman_on_retrieved": rho, "spearman_exact_hungarian": rho_h,
         "top100_overlap": overlap, "n_vox": int(px.size(0)),
         "hungarian_seconds": t_hun, "n_truncated_ligands": n_truncated,
         "exact_full": m_full, "exact_retrieved": m_sub, "vdb": m_vdb,
         "hungarian": m_hun, "ceiling": ceil}
    res.append(r)
    print(f"\n{t}: {n_tot:,} molecules, {n_act:,} actives, {int(px.size(0))} voxels "
          f"({enc:.0f}s exact, {t_ret:.0f}s retrieval, {t_hun:.0f}s assignment)")
    print(f"  retrieved {n_ret:,} ({100*n_ret/n_tot:.1f}%) holding {act_ret}/{n_act} actives "
          f"({100*act_ret/n_act:.0f}%), enrichment {r['retrieval_enrichment']:.2f}x"
          + (f" | {n_truncated} ligand(s) bigger than the pocket" if n_truncated else ""))
    print(f"  {'ranking':<18}{'EF1%':>8}{'EF0.1%':>9}{'EF5%':>8}{'AUC':>8}")
    for nm, m in (("exact_full", m_full), ("exact_retrieved", m_sub), ("vdb (deployed)", m_vdb),
                  ("hungarian", m_hun)):
        print(f"  {nm:<18}{m['ef0.01']:>8.2f}{m['ef0.001']:>9.2f}{m['ef0.05']:>8.2f}"
              f"{m['auc']:>8.3f}")
    print(f"  {'ceiling':<18}{ceil['ef0.01']:>8.2f}{ceil['ef0.001']:>9.2f}"
          f"{ceil['ef0.05']:>8.2f}        (perfect rescoring of the retrieved set)")
    del gd, tg, ag, dg, px
    torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)

print(f"\n{'ranking':<18}{'EF1%':>8}{'EF0.1%':>9}{'EF5%':>8}{'AUC':>8}   (mean over "
      f"{len(res)} targets)")
for nm in ("exact_full", "exact_retrieved", "vdb", "hungarian"):
    g = lambda k_: np.mean([x[nm][k_] for x in res])
    print(f"{nm:<18}{g('ef0.01'):>8.2f}{g('ef0.001'):>9.2f}{g('ef0.05'):>8.2f}{g('auc'):>8.3f}")
ce = lambda k_: np.mean([x["ceiling"][k_] for x in res])
print(f"{'ceiling':<18}{ce('ef0.01'):>8.2f}{ce('ef0.001'):>9.2f}{ce('ef0.05'):>8.2f}")
print(f"\nretrieval keeps {100*np.mean([x['frac_retrieved'] for x in res]):.1f}% of molecules "
      f"and {100*np.mean([x['frac_act_retrieved'] for x in res]):.0f}% of actives "
      f"({np.mean([x['retrieval_enrichment'] for x in res]):.2f}x enrichment)")
print(f"assignment total {sum(x['hungarian_seconds'] for x in res):.0f}s over "
      f"{sum(x['n_retrieved'] for x in res):,} molecules")
print(f"-> {a.out}")
