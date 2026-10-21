# Simpatico — revision final

The model, weights and scoring code behind the reported results. One architecture, one
scoring protocol, two decontamination tiers.

## Layout

    Simpatico/        the package. PocketEncoder architecture; this is the code the
                      official checkpoints were trained and scored with.
    weights/          20 checkpoints, s90 and s30, with arch.json and manifest.tsv
    holdouts/         the split definitions those checkpoints were trained under
    scoring/score.py  the deployed scoring protocol, end to end

## Weights

| benchmark | split | seeds | seed ids |
|---|---|---|---|
| DUD-E | s90 | 3 | 42, 43, 44 |
| DUD-E | s30 | 3 | 42, 43, 44 |
| DEKOIS | s90 | 3 | 42, 43, 44 |
| DEKOIS | s30 | 3 | 42, 43, 44 |
| LIT-PCBA | s90 | 5 | 42, 43, 44, 45, 46 |
| LIT-PCBA | s30 | 3 | 42, 43, 44 |

`weights/arch.json` sits beside them and is **required** — `load_encoders` reads it to pick
the architecture. These checkpoints are `"kind": "pocket"` (PocketEncoder: one node
population, protein atoms within 6 A of any ligand atom, one edge set at 5 A). Without
arch.json the loader falls back to the older voxel ProteinEncoder and the state dict will not
fit. `weights/manifest.tsv` carries the sha256 of every file.

Model weights are committed here deliberately, so this repo is self-contained.

## Scoring

Two paths, both in `scoring/score.py`:

**Deployed** — what the shipped pipeline does, and what the official EF numbers are:
FAISS retrieval over the pocket's voxels keeping the top 2048 neighbours per voxel, a
per-voxel background threshold at the 0.99 quantile, and a score summed over the voxels
that clear it.

**Exact** — every molecule in the library ranked by

    score(m) = mean over pocket voxels of ( max over atoms of m of cos(voxel, atom) )

The two agree closely: across all 29 official checkpoints the deployed path beats exact by
+0.51 EF1% on DUD-E and +0.26 on DEKOIS, and the two orderings never disagree about which
checkpoint is better (Spearman +1.000 on DUD-E, 0 of 36 pairwise flips).

    python scoring/score.py weights/dude_s90_s42_e30.w out.json \
        --screen /path/to/DUDE/screen.csv

### FAISS geometry

This repo ships the **cosine** index. The published numbers were generated with a Euclidean
one. They rank identically: on unit vectors `d^2 = 2(1-cos)`, and max_D, the distance and the
threshold all scale by the same factor of 2, so every molecule's score scales by exactly 2 and
the ordering is unchanged. Verified directly — DEKOIS CDK2 at s90 seed 42 gives EF1% 27.50
under both. The cosine form is kept because it is the geometry the model is trained in.

## Splits

`holdouts/<benchmark>_seq/` gives, per tier, the PDBbind entries withheld. The training set is
the corpus minus the holdout, minus the shared 200-entry validation list, minus 12 blacklisted
multi-model structures. From a 19,434-entry corpus:

| benchmark | s90 training | s30 training |
|---|---|---|
| LIT-PCBA | 18,948 | 16,787 |
| DUD-E | 14,742 | 11,318 |
| DEKOIS | 16,321 | 12,639 |

s60 exists upstream but is not included here.

A standing caveat on the s90 vs s30 comparison: a stricter split removes training data as well
as homology, and at a fixed epoch count that also means fewer gradient steps. DUD-E loses 23.2%
of its training set going to s30 and 40.6% of its EF1%; the two effects are not separated by
these runs.

## Not included

Benchmark libraries (DUD-E, DEKOIS, LIT-PCBA) and the PDBbind corpus. `scoring/score.py` takes
a `--screen` csv pointing at a local copy.
