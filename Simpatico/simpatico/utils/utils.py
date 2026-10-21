import torch
import logging
from os import path
import re
import argparse
from torch_geometric.utils import k_hop_subgraph, to_undirected
from typing import List, Tuple, Optional
from simpatico import config


def to_onehot(value: any, vocabulary: List[any]) -> List[int]:
    """
    Converts a value into a one-hot encoded list based on the given vocabulary.
    Args:
        value (any): The value to be one-hot encoded.
        vocabulary (List[any]): List of all possible values.
    Returns:
        List[int]: One-hot encoded list.
    """
    onehot_list = [0 for _ in vocabulary]
    if value in vocabulary:
        onehot_list[vocabulary.index(value)] = 1
    return onehot_list


def get_k_hop_neighborhoods(atom_idx: int, max_k: int, ei: torch.Tensor) -> List[List]:
    """
    Identifies the k-hop neighborhoods for a given atom.
    Args:
        atom_index (int): Index of the atom of interest.
        max_k (int): Maximum number of hops to consider.
        ei (torch.Tensor): Edge index representing the graph.
    Returns:
        List[List]: List of lists where each list contains the atom indices within each k-hop neighborhood. e.g. [[1-hop], [2-hop], ..., [k-hop]]
    """
    # Track all visited neighbors to avoid repeats, starting with target atom idx
    observed_neighbors = torch.tensor([atom_idx])
    k_hop_neighborhoods = []

    for k_val in range(1, max_k + 1):
        # get all nodes within 'k_val' hops
        k_neighbors = k_hop_subgraph(atom_idx, k_val, ei)[0]

        # Filter out neighbors we have observed before
        k_neighbors = k_neighbors[
            ~torch.any(k_neighbors.unsqueeze(1) == observed_neighbors.unsqueeze(0), 1)
        ]

        observed_neighbors = torch.unique(
            torch.hstack((observed_neighbors, k_neighbors))
        )

        # Remaining unfiltered neighbors must be 'k_val' hops away.
        k_hop_neighborhoods.append(k_neighbors)

    return k_hop_neighborhoods


def get_k_hop_edges(edge_index: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Generates k-hop edges and their corresponding features for a given edge index.
    Args:
        edge_index: Original edge index representing the graph.
        k: Number of hops to consider.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            Updated edge index including k-hop edges, and the corresponding edge features.
    """
    # Vocabulary for edge distances (1-hop, 2-hop, etc.)
    k_hop_vocab = [x + 1 for x in range(k)]

    # Standardize edge index
    edge_index = to_undirected(edge_index.clone())
    # Initialize the final edge index to be constructed
    final_edge_index = torch.tensor([[], []])
    # List to hold edge attributes (one-hot encoded hop distances)
    edge_attr = []

    for atom_idx in edge_index.flatten().unique():
        # For each atom, calculate its k-hop neighborhoods
        for k_i, k_neighborhood in enumerate(
            get_k_hop_neighborhoods(atom_idx.item(), k, edge_index)
        ):
            edge_attr += [
                to_onehot(k_i + 1, k_hop_vocab) for _ in range(k_neighborhood.size(0))
            ]
            # Expand the edge index to include new edges from k-hop neighbors
            final_edge_index = torch.hstack(
                (
                    final_edge_index,
                    torch.vstack(
                        (atom_idx.repeat(k_neighborhood.size(0)), k_neighborhood)
                    ),
                )
            )
    return to_undirected(
        final_edge_index.long(), torch.tensor(edge_attr), reduce="mean"
    )


def get_mol2_coords(input_file) -> torch.Tensor:
    """
    Retrieves coordinate values from a .mol2 file.
    Args:
        input_file (str): path to .mol2 file
    Returns:
        (torch.Tensor): (N,3)-shaped tensor of XYZ coordinates.
    """
    coord_pattern = re.compile(
        r"^\s*\d+\s+\S+\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)"
    )
    coord_list = []

    with open(input_file) as file_in:
        for line in file_in:
            match = coord_pattern.match(line)
            if match:
                xyz = [float(v) for v in match.groups()]
                coord_list.append(xyz)

    return torch.tensor(coord_list)

def get_logger(output_file, log_config = None):
    if log_config is None:
        log_config = {'level': logging.INFO,
                    'format': '%(asctime)s |--| %(message)s',
                    'force': True}

    log_config['filename'] = output_file

    logging.basicConfig(**log_config)
    log = logging.getLogger(__name__)
    return log
