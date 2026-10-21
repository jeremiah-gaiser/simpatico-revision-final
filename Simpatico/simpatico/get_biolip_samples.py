import argparse
import os
import pickle
import torch
import sys
import re
from simpatico.utils.pdb_utils import pdb2pyg, extract_ligands, trim_protein_graph, pdb_to_bio
from collections import defaultdict
from Bio.PDB import is_aa
import warnings
from rdkit import RDLogger
import warnings
warnings.filterwarnings("ignore")



def add_arguments(parser):
    parser.add_argument(
        "biolip_file",
        type=str,
        help="Path to BioLiP.txt file.",
    )
    parser.add_argument(
        "pdb_dir",
        type=str,
        help="Path to directory containing pdb files corresponding to BioLiP.txt entries.",
    )
    # sets parser's main function to the main function in this script
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory where protein and ligand sample graphs should be stored.",
    )
    parser.set_defaults(main=main)

def get_res_coords(structure):
    res_coords = {}

    for residue in structure.get_residues():
        if is_aa(residue) == False:
            continue
        residue_key = residue.parent.id + str(residue.id[1])
        res_atom_coords = []

        for atom in residue.get_atoms():
            res_atom_coords.append(atom.get_coord())
        
        res_coords[residue_key] = torch.tensor(res_atom_coords).mean(0)

    return res_coords

def get_binding_site_coords(res_coords, res_keys):
    mean = torch.zeros(3).float()
    
    try:
        for res in res_keys:
            mean += res_coords[res] / len(res_keys)
    except:
        return None

    return mean 

def get_binding_site_residues(line):
    line_content = re.split(r'\t', line) 
    chain = line_content[1]
    residues = re.split(r'\s+', line_content[7])
    residue_numbers = [chain+x[1:] for x in residues]
    return residue_numbers

def get_proximal_ligand(ligand_collection, site_coords):
    D = [torch.cdist(site_coords.unsqueeze(0), l.pos).mean() for l in ligand_collection]
    argmin = D.index(min(D))
    print(argmin)

    return ligand_collection[argmin]

def main(args):
    pdb_dir = args.pdb_dir
    out_dir = args.output_dir

    if pdb_dir[-1] != '/':
        pdb_dir += '/'

    if out_dir[-1] != '/':
        out_dir += '/'

    with open(args.biolip_file, 'r') as biolip_in:
        prev_pdb_id = ''
        prev_ligand_id = ''

        for l_i, line in enumerate(biolip_in):
            line_content = re.split(r"\t", line.strip())

            pdb_id = line_content[0]
            ligand_id = line_content[4] 

            if prev_pdb_id != pdb_id:
                pdb_file = pdb_dir + pdb_id + '.pdb'

                if os.path.exists(pdb_file) == False:
                    pdb_file = pdb_dir + pdb_id + '.cif'
                
                print(pdb_file)
                protein_structure = pdb_to_bio(pdb_file)
                protein_structure.id = pdb_file

                protein_graph = pdb2pyg(protein_structure)
                res_coords = get_res_coords(protein_structure) 

            if prev_ligand_id != ligand_id:
                ligand_collection = extract_ligands(protein_structure, ligand_id)

            prev_pdb_id = pdb_id 
            prev_ligand_id = ligand_id

            if len(ligand_collection) == 0:
                print('No viable ligands.')
                continue

            if len(ligand_collection) < 2:
                ligand_g = ligand_collection[0]
            else:
                site_numbers = get_binding_site_residues(line)
                site_coords = get_binding_site_coords(res_coords, site_numbers)

                if site_coords is None:
                    print('FAIL')
                    continue

                ligand_g = get_proximal_ligand(ligand_collection, site_coords)

            pocket_g = trim_protein_graph(protein_graph.clone(), ligand_g.pos)
            pocket_g.name = ligand_g.name

            outfile = f"{out_dir}{ligand_g.name}_%s.pyg"

            with open(outfile % 'ligand', 'wb') as ligand_out:
                pickle.dump(ligand_g, ligand_out)

            with open(outfile % 'pocket', 'wb') as pocket_out:
                pickle.dump(pocket_g, pocket_out)