from typing import List
from torch_geometric.nn import radius
import torch


def get_proximal_atom_masks(
    protein_pos: torch.Tensor, ligand_pos: torch.Tensor, r: float = 4
) -> List[torch.Tensor]:
    """
    Generate per-atom masks for protein and ligand graphs indicating which are involved in interactions.
    Args:
        protein_pos (torch.Tensor): position values of protein graph.
        ligand_pos (torch.Tensor): position values of ligand graph.
        r (float, optional): threshold distance for interactivity (default = 4).
    Returns:
        (torch.Tensor, torch.Tensor): boolean masks for protein and ligand graphs.
    """
    protein_mask = torch.zeros(protein_pos.size(0)).bool()
    ligand_mask = torch.zeros(ligand_pos.size(0)).bool()

    interaction_index = radius(protein_pos, ligand_pos, r)

    protein_mask[interaction_index[1].unique()] = True
    ligand_mask[interaction_index[0].unique()] = True

    return protein_mask, ligand_mask


def _sphere_points(n: int, device, dtype) -> torch.Tensor:
    """`n` roughly-equidistant points on the unit sphere (Fibonacci / golden spiral)."""
    i = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = torch.pi * (1 + 5**0.5) * i
    return torch.stack(
        [theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()], dim=1
    )


def surface_mask(
    pos: torch.Tensor,
    batch: torch.Tensor = None,
    probe: float = 1.4,
    atom_radius: float = 1.8,
    n_points: int = 92,
    min_exposure: float = 0.01,
    max_neighbors: int = 160,
    edge_chunk: int = 20000,
) -> torch.Tensor:
    """Shrake-Rupley solvent exposure, reduced to a boolean per-atom mask.

    A uniform heavy-atom radius is used rather than per-element radii: the output is a binary
    exposed/buried call, not a surface AREA, and that call is insensitive to the 1.55-1.9 A
    spread of protein heavy-atom radii. Uniformity also means the mask depends only on `pos`,
    so it is identical whichever feature representation the graph carries.

    Note this is the local Shrake-Rupley test -- a point is exposed if no OTHER atom covers it.
    It does not ask whether a probe could reach that point from bulk solvent, which is exactly
    what we want here: the atoms lining a binding pocket are the ones being kept, and a
    reachability test run from the outside would have to be told that the pocket counts as
    outside.

    Args:
        pos: (N, 3) coordinates.
        batch: (N,) graph assignment; occlusion is only ever computed within a graph.
        probe: solvent probe radius.
        atom_radius: uniform heavy-atom vdW radius.
        n_points: sphere points per atom; the exposure resolution is 1/n_points.
        min_exposure: fraction of points that must be unoccluded to call an atom surface.
            The default 0.01 is "at least one of 92 points", i.e. strictly nonzero SASA.
    Returns:
        (N,) boolean mask, True for surface atoms.
    """
    device, dtype = pos.device, pos.dtype
    N = pos.size(0)
    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    R = atom_radius + probe
    u = _sphere_points(n_points, device, dtype) * R

    # Two atoms can only occlude each other's probe spheres within 2R.
    row, col = radius(pos, pos, 2 * R, batch, batch, max_num_neighbors=max_neighbors)
    keep = row != col
    row, col = row[keep], col[keep]

    # Occlusion count per (atom, sphere point). Accumulated as float because index_add_ has no
    # boolean form; any nonzero entry means the point is covered.
    occ = torch.zeros(N, n_points, device=device, dtype=dtype)
    for s in range(0, row.numel(), edge_chunk):
        r, c = row[s : s + edge_chunk], col[s : s + edge_chunk]
        d2 = (pos[r].unsqueeze(1) + u.unsqueeze(0) - pos[c].unsqueeze(1)).pow(2).sum(-1)
        occ.index_add_(0, r, (d2 < R * R).to(dtype))

    exposure = (occ == 0).to(dtype).mean(dim=1)
    return exposure > min_exposure
