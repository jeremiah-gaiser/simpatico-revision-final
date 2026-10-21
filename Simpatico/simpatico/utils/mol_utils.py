import os
import pickle
import re
import sys
from copy import deepcopy
from os import path
from typing import List, Optional, Tuple

import numpy as np
import torch
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph, to_undirected

from simpatico import config
from simpatico.utils.utils import get_k_hop_edges, to_onehot


def get_mol_atom_features(m: Mol, atom_vocab: List[str]) -> torch.Tensor:
    """
    Generates one-hot feature tensor to reflect atomic symbol of each atom.
    Args:
        m (Mol): rdkit Molecule object.
        atom_vocab (List[str]): List of possible atomic symbols.
    Returns:
       torch.Tensor: (M,N) shaped pytorch tensor where M is number of atoms and N is length of atom vocab.
    """
    atom_count = m.GetNumAtoms()

    # placeholder for one-hot rows
    x = [0 for _ in range(atom_count)]

    for atom in m.GetAtoms():
        atom_idx = atom.GetIdx()
        # produce list of 0s except for 1 at index corresponding to atomic symbol
        onehot_row = to_onehot(atom.GetSymbol(), atom_vocab)
        x[atom_idx] = onehot_row

    return torch.tensor(x)


def get_mol_pos(m: Mol) -> torch.Tensor:
    """
    Get xyz coordinates for each atom in rdkit Molecule object.
    Args:
        m (Mol): rdkit Molecule object.
    Returns:
        torch.Tensor: (M,3) shaped pytorch tensor.
    """
    atom_count = m.GetNumAtoms()
    # retrieve 3D coordinates describing molecular conformer
    conformer = m.GetConformer()
    # placeholder for positional data
    pos = [0 for _ in range(atom_count)]

    for atom in m.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_pos = conformer.GetAtomPosition(atom_idx)
        pos[atom_idx] = [atom_pos.x, atom_pos.y, atom_pos.z]

    return torch.tensor(pos)


def get_mol_edges(m: Mol, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get pyg graph style molecular edge index.
    Args:
        m (Mol): rdkit Molecule object.
        k (int): featured edges will be generated for any atoms k bonds away from each other.
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]:
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    """
    edge_index = [[], []]

    # get covalent bonds from list of (from_idx, to_idx) bond descriptions
    for b in m.GetBonds():
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[1].append(b.GetEndAtomIdx())

    # 'get_k_hop_edges' accepts an edge index pytorch tensor
    edge_index_tensor = torch.tensor(edge_index)

    # generate edges and edge features for atoms k covalent bonds away from each other
    final_edge_index, final_edge_attr = get_k_hop_edges(edge_index_tensor)

    return final_edge_index, final_edge_attr


def get_H_counts(
    x: torch.Tensor, edge_index: torch.Tensor, atom_vocab: List[str]
) -> torch.Tensor:
    """
    Generates one-hot feature tensor to reflect number of hydrogens attached to each atom.
    Args:
        x (torch.Tensor): one-hot feature vector reflecting atomic symbol of each atom for molecular graph
        edge_index (torch.Tensor): pyg style edge index reflecting covalent bonds (k-hop edges must be trimmed)
        atom_vocab (List[str]): List of possible atomic symbols.
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]:
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    """
    # one-hot vocab for possible number of hydrogens connected to single atoms
    H_count_vocab = [v for v in range(4)]

    # initialize H count feature vector
    # for each row in x, generate row of zeros equal in length to H_count_vocab
    H_count_features = torch.zeros(x.size(0), len(H_count_vocab))

    # use undirected_graph to remove any duplicate edges and standardize edge index
    ei = to_undirected(edge_index)
    H_idx = atom_vocab.index("H")

    # get indices of hydrogen atoms in x
    H_atom_index = torch.where(x[:, H_idx] == 1)[0]

    # get indices of all edges where hydrogen is the sink node (heavy-to-hydrogen edges)
    H_node_sinks = torch.where(
        # produce heavy-to-hydrogen mask by mapping each sink value as follows:
        # if idx value is in list of hydrogen indices (H_atom_index), return True
        # otherwise return False
        (ei[1].unsqueeze(1) == H_atom_index.unsqueeze(0)).any(dim=1)
    )[0]

    heavy_to_hydrogen_edges = ei[:, H_node_sinks]

    # instance count of heavy atom index in heavy_to_hydrogen_edges reflects its H count
    H_neighbors, H_counts = heavy_to_hydrogen_edges[0].unique(return_counts=True)

    # update one hot to reflect number of hydrogens connected to each heavy atom
    H_counts[H_counts > 3] = 3
    H_count_features[H_neighbors.long(), H_counts.long()] = 1
    # For atoms not connect to a hydrogen, set first index of onehot to 1 to indicate zero Hs
    H_count_features[H_count_features.sum(1) == 0, 0] = 1
    return H_count_features


def get_xyz_from_file(input_file: str) -> torch.Tensor:
    """
    Generates a tensor of x,y,z coordinate values from molecular structure  or .csv file.

    Args:
        input_file (str): path to file (.csv, .txt, .mol2, .sdf)
    Returns:
        (torch.Tensor): float-valued tensor of shape (N,3)
    """
    _, filetype = path.splitext(input_file)
    xyz_coords = []

    if filetype == '.pyg':
        with open(input_file, 'rb') as graph_in:
            g = pickle.load(graph_in)
            return g.pos

    elif filetype == '.mol2':
        xyz_coords = get_pos_from_mol2(input_file)

    elif filetype in config["molecule_filetypes"]:
        mols = molfile2rdkit(input_file)
        for m in mols:
            m_pos = get_mol_pos(m)
            xyz_coords.append(m_pos)

        xyz_coords = torch.vstack(xyz_coords)
    else:
        with open(input_file) as pos_in:
            for line in pos_in:
                line_content = [float(x.strip()) for x in line.split(",")]
                xyz_coords.append(line_content)

        xyz_coords = torch.tensor(xyz_coords)
    return xyz_coords


def mol2pyg(
    m: Mol,
    ignore_pos: bool = False,
    removeHs: bool = True,
    get_scaffold=False,
    source_idx=None,
) -> Optional[Data]:
    """
    Converts an RDKit Mol object into a PyG Data object representing the molecular graph.
    Args:
        m (Mol): RDKit Molecule object.
        removeHs (bool): If True, hydrogen atoms are removed from the final graph.
    Returns:
        Optional[Data]: PyG Data object containing the graph representation of the molecule, or None if conversion fails.
    """
    standardizer = Standardizer()
    RDLogger.DisableLog("rdApp.*")

    # If we cannot standardize molecule, we won't be able to extract the scaffold
    try:
        m = standardizer.standardize(m)
    except Exception as e:
        get_scaffold = False

    # Attempt to add hydrogen atoms so that we are working with a standardized molecule going forward
    try:
        m = AddHs(m)
    except:
        print("Could not add hydrogens")
        return None

    if get_scaffold:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(m)
            scaffold_atoms = m.GetSubstructMatch(scaffold)

            if len(scaffold_atoms) == 0:
                get_scaffold = False
            else:
                scaffold_atoms = torch.tensor(list(scaffold_atoms))
        except Exception as e:
            print(e)
            get_scaffold = False

    if ignore_pos is False:
        # Get the 3D coordinates of atoms in the molecule
        pos = get_mol_pos(m)


    mol_atom_vocab = config.get("mol_atom_vocab")

    # Generate one-hot encoded atom features
    atom_species_onehots = get_mol_atom_features(m, mol_atom_vocab)

    if get_scaffold:
        scaffold_mask = torch.zeros(atom_species_onehots.size(0)).bool()
        scaffold_mask[scaffold_atoms] = True

    # Generate edge indices and edge attributes for the molecular graph
    edge_index, edge_attr = get_mol_edges(m)

    if edge_index.size(1) == 0:
        return None

    # Identify covalent bonds (edges where k = 1)
    covalent_index = torch.where(edge_attr[:, 0] == 1)[0]
    covalent_edge_index = edge_index[:, covalent_index]

    # Generate features representing the number of hydrogens attached to each atom
    H_count_features = get_H_counts(
        atom_species_onehots, covalent_edge_index, mol_atom_vocab
    )

    # Concatenate the hydrogen count features to the atomc species onehots
    x = torch.hstack((atom_species_onehots, H_count_features))

    if removeHs:
        H_idx = mol_atom_vocab.index("H")
        # Get indices of non-hydrogen (heavy) atoms
        heavy_atom_index = torch.where(x[:, H_idx] != 1)[0].long()

        try:
            # Remove hydrogen-adjacent edges
            edge_index, edge_attr = subgraph(
                heavy_atom_index, edge_index, edge_attr, relabel_nodes=True
            )
            x = x[heavy_atom_index]
        except:
            return None

        if get_scaffold:
            scaffold_mask = scaffold_mask[heavy_atom_index]

        if ignore_pos is False:
            pos = pos[heavy_atom_index]

    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce="mean")

    # Create a PyG Data object representing the molecular graph
    mol_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if get_scaffold:
        mol_graph.scaffold = scaffold_mask

    if ignore_pos is False:
        mol_graph.pos = pos

    if source_idx is not None:
        mol_graph.source_idx = torch.zeros(len(mol_graph.x)).fill_(source_idx).long()

    return mol_graph

def get_pos_from_mol2(input_file):
    xyz = []

    with open(input_file) as mol2_in:
        pos_block = False
        for line in mol2_in:
            if line[:13] == '@<TRIPOS>ATOM':
                pos_block = True
                continue

            if pos_block and line[0] == '@':
                pos_block = False

            if pos_block == False:
                continue

            line_content = re.split(r'\s+', line.strip())

            if line_content[1] == 'H':
                continue

            xyz.append([float(x.strip()) for x in line_content[2:5]])

    return torch.tensor(xyz)

def molfile2rdkit(m_file, header=False):
    # Extract the filename and filetype from the input file path
    _, filetype = os.path.splitext(m_file)

    # Use appropriate RDKit method for generating Molecule object from file
    if filetype == ".sdf":
        mols = Chem.SDMolSupplier(m_file, sanitize=False)

    elif filetype == ".pdb":
        mols = [Chem.MolFromPDBFile(m_file, sanitize=False)]

    elif filetype in [".ism", ".smi", ".cxsmiles"]:
        mols = Chem.SmilesMolSupplier(m_file, sanitize=False, titleLine=header)

    elif filetype == ".mol2":
        mols = [Chem.MolFromMol2File(m_file, sanitize=False)]

    return mols

def is_smile_file(m_file):
    _, filetype = os.path.splitext(m_file)
    if filetype in [".ism", ".smi", ".cxsmiles"]:
        return True
    return False


def molfile2pyg(
    m_file: str,
    k: int = 3,
    ignore_pos=False,
    get_scaffold: bool = False,
) -> Optional[Batch]:
    """
    Converts a molecular file (e.g., SDF, PDB) into a batch of molecular PyG graphs.
    Args:
        m_file (str): Path to the molecular file.
        get_pos (bool): If True, 3D coordinates of atoms are included in the graph.
        k (int): Maximum number of covalent bonds between two atoms for which edges are generated.
    Returns:
        Optional[Batch]: A batch of PyG Data objects, or None if conversion fails.
    """
    if m_file[-3:] == 'pyg':
        return torch.load(m_file, weights_only=False)

    mols = molfile2rdkit(m_file)
    smiles = []

    filename = '.'.join(m_file.split('/')[-1].split('.')[:-1])

    if is_smile_file(m_file):
        ignore_pos = True

    # List to store individual PyG graph objects
    mol_batch = []
    s_i = 0

    for m in mols:
        if m is None:
            mg = None
        else:
            mg = mol2pyg(
                m,
                ignore_pos,
                get_scaffold=get_scaffold,
            )
        if mg is None:
            # Create dummy molecule to maintain proper indexing
            mg = mol2pyg(Chem.MolFromSmiles('C'), ignore_pos, get_scaffold=get_scaffold)
            mg.x[mg.x != 0] = 0
            m_name = filename + f"_{s_i}"
        else:
            if m.HasProp("_Name"):
                m_name = m.GetProp("_Name")
            else:
                m_name = filename + f"_{s_i}"

        mg.name = m_name
        mol_batch.append(mg)
        try:
            # Try generating SMILES with default settings
            smile_string = Chem.MolToSmiles(m)
        except:
            smile_string = 'NA'

        smiles.append(smile_string)
        s_i += 1

    if len(mol_batch) == 0:
        # Return None if no valid graphs were generated
        return None, None

    # Create a batch of PyG Data objects
    pyg_batch = Batch.from_data_list(mol_batch)
    pyg_batch.source = m_file
    return pyg_batch, smiles
