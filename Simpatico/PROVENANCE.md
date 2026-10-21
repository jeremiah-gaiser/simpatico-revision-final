# Provenance

Copy of the Simpatico model code **as actually used by every run in this repository**.

    source   /home/jgaiser/2026/Final/simpaticoV2
    remote   git@github.com:jeremiah-gaiser/simpatico-revision.git
    branch   fix/pocket-definition-and-nmr-parsing
    commit   2ddb36223af6ef18edbf656b8e71ae4a8d16c4ad
    date     2026-09-25 08:35:53 -0700
    copied   2026-09-25T10:24:02-07:00

Copied without `.git`, so this is a snapshot rather than a clone.

## Why this exists

A previous copy in this directory was a clone of TravisWheelerLab/Simpatico at `8b1959b`
(2026-05-21, branch `master`). Nothing used it: every worker script sets
`PYTHONPATH=/home/jgaiser/2026/Final/simpaticoV2`. It lacked all five changes the results here
depend on, so a stale copy that looked like the working code was worse than none.

## Commits this snapshot carries beyond that stale copy

    2ddb362 Give the two contrastive losses independent temperatures
    c095959 pdb2pyg: read only the first model, and record the pocket coordinates
    53ca26a Build the contrastive-loss masks in one buffer
    d908cb9 Fix the protein path in evaluate.py
    a007d63 Remove the dead randomized-pocket path from training

The temperature keys (`temperature_struct`, `temperature_hard`) read by
`runs/tgrid/*/config.json` and `runs/tfinal/*/config.json` exist only from `2ddb362`, so
without this snapshot those configs could not be reproduced.
