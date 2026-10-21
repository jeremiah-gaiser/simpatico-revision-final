# models/__init__.py
import json
import os
import importlib.resources as pkg_resources

# Define the path to the JSON file
# mol_hparam_path = os.path.join(
#     os.path.dirname(__file__), "molecule_encoder/MolEncoder.json"
# )
# protein_hparam_path = os.path.join(
#     os.path.dirname(__file__), "protein_encoder/ProteinEncoder.json"
# )

# with open(mol_hparam_path, "r") as file:
#     MolEncoderDefaults = json.load(file)

# with open(protein_hparam_path, "r") as file:
#     ProteinEncoderDefaults = json.load(file)

with pkg_resources.open_text(
    "simpatico.models.molecule_encoder", "MolEncoder.json"
) as f:
    MolEncoderDefaults = json.load(f)

with pkg_resources.open_text(
    "simpatico.models.protein_encoder", "ProteinEncoder.json"
) as f:
    ProteinEncoderDefaults = json.load(f)


def load_encoders(weights_file, device="cuda", eval_mode=True):
    """Build both encoders from a checkpoint, honouring the arch.json written beside it.

    `surface_only` (ProteinEncoder) and `max_hops` (MolEncoder) change what graph the model
    sees but no parameter shape, so `load_state_dict` accepts a checkpoint trained with them
    into a default-constructed encoder without error -- and then evaluates a different model
    than the one that was trained. train.py writes weights/arch.json next to the weights;
    this reads it, and falls back to library defaults when it is absent (every checkpoint
    predating the margin-semantic branch).

    Args:
        weights_file: path to a `[protein_state_dict, mol_state_dict]` .w file.
        device: torch device or string.
        eval_mode: call .eval() on both (turns off DropEdge and dropout).
    Returns:
        (ProteinEncoder, MolEncoder)
    """
    import torch
    from pathlib import Path

    from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
    from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder

    wf = Path(weights_file)
    arch = {}
    for cand in (wf.parent / "arch.json", wf.parent.parent / "weights" / "arch.json"):
        if cand.exists():
            with open(cand) as f:
                arch = json.load(f)
            break

    # "kind" names the pocket architecture rather than configuring it, so it is popped before
    # the rest are passed as kwargs. "voxel" is ProteinEncoder's two-population graph (voxel
    # nodes plus atom context, three edge generators); "pocket" is PocketEncoder's single
    # population with one edge set. Absent means voxel -- every checkpoint predating the
    # simple-pocket branch.
    prot_arch = dict(arch.get("protein", {}))
    kind = prot_arch.pop("kind", "voxel")
    if kind == "pocket":
        from simpatico.models.protein_encoder.PocketEncoder import PocketEncoder

        pe = PocketEncoder(**prot_arch).to(device)
    elif kind == "voxel":
        pe = ProteinEncoder(**prot_arch).to(device)
    else:
        raise ValueError(f"unknown protein encoder kind {kind!r} in arch.json")
    me = MolEncoder(**arch.get("molecule", {})).to(device)
    pw, mw = torch.load(weights_file, map_location=device)
    pe.load_state_dict(pw)
    me.load_state_dict(mw)
    if eval_mode:
        pe.eval()
        me.eval()
    return pe, me
