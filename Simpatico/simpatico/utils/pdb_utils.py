import re
import tempfile
from os import path
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius
from simpatico import config
from simpatico.utils.utils import to_onehot
from simpatico.utils.mol_utils import molfile2pyg, mol2pyg
from typing import List, Tuple, Optional
from collections import defaultdict
from Bio.PDB import PDBParser, MMCIFParser, is_aa, PDBIO
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity



def trim_protein_graph(protein_graph, target_pos, r=15.0):
    """Keep protein atoms within `r` of the target.

    r=15.0, not 12.5. ProteinEncoder masks to atoms within 6 A of the ligand and then gathers
    context within 8 A of those, so its receptive field reaches ~14 A; a 12.5 A trim clips it
    and yields DIFFERENT encoder output than the untrimmed graph the eval path builds. Verified
    on 10gs/11gs/13gs/1a30: at 15 A the voxel embeddings are identical to untrimmed, at 12.5 A
    they are not.

    `r` was also previously ignored -- the radius call hardcoded 12.5, so the parameter had no
    effect."""
    proximal_atoms = radius(target_pos, protein_graph.pos, r)[0].unique()

    protein_graph.x = protein_graph.x[proximal_atoms]
    protein_graph.pos = protein_graph.pos[proximal_atoms]
    protein_graph.residue = protein_graph.residue[proximal_atoms]
    protein_graph.chain = protein_graph.chain[proximal_atoms]
    return protein_graph

def get_atom_features(atom) -> Tuple[List[float], List[float]]:
    """
    Takes a Bio.PDB.Atom object from a Bio.PDB.Structure and returns one-hot feature and positional tensors for PyG graphs.
    Args:
        atom (Bio.PDB.Atom): atom object from structure.
    Returns:
        Tuple[List[float], List[float]]: Per-atom one-hot feature and positional tensors
    """
    atom_vocab = config.get("atom_vocab")
    res_vocab = config.get("res_vocab")
    
    onehot = []
    pos = atom.get_coord()

    residue = atom.parent

    # Get one-hot feature vectors given corresponding feature vocab
    onehot += to_onehot(atom.get_name(), atom_vocab)
    onehot += to_onehot(residue.get_resname(), res_vocab)

    chain_val = residue.parent.id
    residue_number = residue.id[1]

    return onehot, pos, chain_val, residue_number

def pdb2pyg(pdb_path, ligand_pos=None, pocket_coords=None) -> Data:
    """
    Converts PDB file to PyG graph
    Args:
        pdb_path (str): path to PDB file to be converted
    Returns:
        Data: PyG graph object
    """
    if isinstance(pdb_path, str):
        pdb_structure = pdb_to_bio(pdb_path)
    else:
        pdb_structure = pdb_path
        pdb_path = pdb_structure.id

    pdb_name = path.splitext(path.basename(pdb_path))[0]

    graph_x = []
    graph_pos = []
    chain_vals = []
    res_numbers = []

    # Only the first model. NMR ensembles carry up to 21 MODEL records, and iterating the
    # whole structure concatenates every conformation into one point cloud -- 6h8c becomes
    # 16,680 atoms instead of 1,097, with 17x the ligand contacts.
    if hasattr(pdb_structure, "get_models"):
        models = list(pdb_structure.get_models())
        atom_source = models[0] if models else pdb_structure
    else:
        atom_source = pdb_structure

    for atom in atom_source.get_atoms():
        if atom.element == 'H':
            continue
        if is_aa(atom.get_parent()) == False:
            continue

        x, pos, chain, res_number = get_atom_features(atom)

        graph_x.append(x)
        graph_pos.append(list(pos))
        chain_vals.append(chain)
        res_numbers.append(res_number)

    unique_chain_vals = list(set(chain_vals))
    chain_vals = [unique_chain_vals.index(x) for x in chain_vals]

    g = Data(
        x=torch.tensor(graph_x),
        pos=torch.tensor(graph_pos),
        residue=torch.tensor(res_numbers).int(),
        chain=torch.tensor(chain_vals).int(),
        name=pdb_name,
        source=pdb_path,
    )

    if ligand_pos is not None:
        g = trim_protein_graph(g, ligand_pos)

    if pocket_coords is not None:
        # Record the coordinates the pocket is defined by rather than a mask over them.
        # ProteinEncoder.forward derives the pocket from these at embed time, so a cached
        # graph carries everything needed to reproduce the pocket it was built for.
        g.pocket_coords = torch.as_tensor(pocket_coords).float()

    return g

def pdb_to_bio(pdb_file):
    pdb_filetype = path.splitext(path.basename(pdb_file))[1]

    if pdb_filetype == '.pdb':
        parser = PDBParser()
    else:
        parser = MMCIFParser()

    # Use the filename (without extension) as graph name
    pdb_name = path.splitext(path.basename(pdb_file))[0]
    pdb_data = parser.get_structure(pdb_name, pdb_file)

    try:
        pdb_data = pdb_data[0]
    except:
        return None

    
    return pdb_data

def extract_ligands(pdb_path: str, ligand_id: str):
    if isinstance(pdb_path, str):
        pdb_structure = pdb_to_bio(pdb_path)
        if pdb_structure is None:
            return [] 
    else:
        pdb_structure = pdb_path
        pdb_path = pdb_structure.id

    pdb_filename = pdb_path.split('/')[-1].split('.')[0]

    ligand_graphs = []

    for r_i, res in enumerate(pdb_structure.get_residues()):
        if is_aa(res):
            continue
           
        if res.get_resname() == ligand_id:
            mol = residue_to_rdkit(res)

            if mol is None:
                continue

            try:
                g = mol2pyg(mol)
            except:
                g = None

            if g is not None:
                g.name = f'{pdb_filename}_{ligand_id}_{res.parent.id}_{res.id[1]}_{r_i}'
                ligand_graphs.append(g)
                    
    return ligand_graphs

def residue_to_rdkit(residue):
    no_carbon = True 
    mol = Chem.RWMol()
    atom_map = {}

    # Add atoms
    try:
        for atom in residue.get_atoms():
            atom_element = atom.element.capitalize()
            if atom_element == 'C':
                no_carbon = False 
            idx = mol.AddAtom(Chem.Atom(atom_element))
            atom_map[atom] = idx
    except:
        print('Weird atoms found')
        return None
    
    if no_carbon:
        'No carbon present in ligand.'
        return None

    # Add coordinates
    conf = Chem.Conformer(len(atom_map))
    for atom, idx in atom_map.items():
        x, y, z = atom.coord
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    try:
        DetermineConnectivity(mol)
    except:
        print("could not assign bonds.")
        return None
   
    # Guess bonds
    # rdDetermineBonds.DetermineConnectivity()
    # rdmolops.SanitizeMol(mol, catchErrors=True)

    return mol


