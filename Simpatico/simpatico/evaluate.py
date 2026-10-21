# scripts/train.py
import argparse
import logging
import os
import pickle
import sys
import traceback
from glob import glob
from os import path
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius

from simpatico import config
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.utils.app_utils import get_encoder
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
    handle_no_overwrite,
)
from simpatico.utils.mol_utils import get_xyz_from_file, molfile2pyg
from simpatico.utils.pdb_utils import extract_ligands, pdb2pyg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file.\n"
        "- Protein: .pdb file\n"
        "- Small molecule: .smi, .ism, .sdf, or .pdb file\n"
        "- Batch: .csv file with one item per line:\n"
        "    For proteins:\n"
        "      <protein_structure_path>, <pocket_spec_path>\n"
        "      ..."
        "  For molecules:\n"
        "      <molecule_structure_path>"
        "      ...",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to directory for outputting embed file."
    )
    parser.add_argument(
        "-w",
        "--weights-file",
        default=config["default_weights_path"],
        help="Non-default weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (gpu or cpu)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output files. For example, with `--suffix _test`, output.pyg becomes output_test.pyg",
    )
    parser.add_argument(
        "--pocket-coordinates",
        type=str,
        default=None,
        help="File specifying pocket coordinates (.csv or any compatible small-molecule file)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip evaluation if output file already exists.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--protein",
        action="store_true",
        help="Indicates protein evaluation.",
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Indicates molecule evaluation.",
    )
    parser.add_argument(
        '--name-depth',
        type=int,
        default=1
    )
    parser.add_argument(
        '--pocket-id',
        type=str,
        help='ligand ID from PDB file to use as pocket location'
    )
    parser.add_argument(
        '--pocket-id-index',
        type=int,
    )

    parser.set_defaults(main=main)
    return parser

def get_input_list(args) -> list[list[str]]:
    """
    Gather protein/molecule structure files and produce list for evaluation.

    Args:
        args (ArgumentParser): parsed script arguments
    Returns:
        (list[list[str]]): list of lists of paths to structural files
    """
    input_file = args.input_file

    _, input_filetype = path.splitext(args.input_file)

    # store input file paths here
    input_list = []

    # if input is a batch file, iterate through and store paths in `input_list`
    if input_filetype in [".txt", ".csv"]:
        with open(input_file, "r") as input_in:
            for line in input_in:
                line_content = [x.strip() for x in line.split(",")]
                input_list.append(line_content)
    # if input is an individual structure file, store in `input_list` to be compatible with remaining script.
    else:
        input_list = [[input_file, None]]

    check_files(input_list)

    return input_list


def check_files(input_list: list[list[str]]) -> None:
    """
    Check that all files included in input list produced by `get_input_list` exist. Raise FileNotFoundError if a file is missing.
    Args:
        args (ArgumentParser): parsed script arguments
    Returns:
        (None): list of lists of paths to structural files
    """
    for row in input_list:
        for input_file in row:
            if input_file is None:
                continue

            if not path.exists(input_file):
                raise FileNotFoundError(f"Input file does not exist: {input_file}")


def get_input_data_loader(input_line: list[str], args: argparse.ArgumentParser):
    """
    Prepare item from input list for evaluation, according to logic reserved for different input types.
    Args:
        input_line (list[str]): line from input list produced by `get_input_list`
        args (ArgumentParser): script arguments
    Returns:
        (DataLoader): data loader object containing data for eval.
    """
    structure_file = input_line[0]

    structure_is_pyg = False
    if structure_file.split('.')[-1] == 'pyg':
        structure_is_pyg = True

    if len(input_line) > 1:
        pocket_data = input_line[1]
    else:
        pocket_data = None

    pocket_spec = None

    if args.protein:
        # if single protein structural file is input, path to pocket coordinate file is supplied in args
        if args.pocket_id:
            pocket_ligand = extract_ligands(structure_file, args.pocket_id)
            if len(pocket_ligand) > 1:
                if args.pocket_id_index is None:
                    sys.exit(f'{len(pocket_ligand)} ligands found with specified id. Please provide an index (--pocket-id-index [1-{len(pocket_ligand)}])')
                else:
                    pocket_ligand = pocket_ligand[args.pocket_id_index-1]
            else:
                pocket_ligand = pocket_ligand[0]

            pocket_spec = pocket_ligand.pos
        else:
            pocket_spec_file = pocket_data or args.pocket_coordinates
            if pocket_spec_file is not None:
                pocket_spec = get_xyz_from_file(pocket_spec_file)

    if structure_is_pyg:
        print(structure_file)
        input_g = torch.load(structure_file, weights_only=False)
        # input_g = pickle.load(open(structure_file,'rb'))

        # a graph cached by pdb2pyg carries the coordinates its pocket was defined by
        if args.protein and pocket_spec is None:
            pocket_spec = getattr(input_g, "pocket_coords", None)
    elif args.protein:
        if pocket_spec is None:
            sys.exit(
                f"No pocket specified for {structure_file}. Supply one with "
                "--pocket-coordinates, --pocket-id, or a second column in the input list."
            )
        input_g = pdb2pyg(structure_file, pocket_coords=pocket_spec)
        input_g = Batch.from_data_list([input_g])
    elif args.molecule:
        input_g,_ = molfile2pyg(structure_file)

    input_data_loader = DataLoader(input_g, batch_size=1024, shuffle=False)
    input_data_loader.source_file = structure_file
    input_data_loader.pocket_spec = pocket_spec

    return input_data_loader


def evaluate_data(input_data_loader, outfile, encoder, args):
    """
    Produce embeddings from data.
    Args:
        input_data_loader (DataLoader): input data loader.
        outfile (str): path to output file string.
        encoder (MolEncoder | ProteinEncoder): encoding model to generate embeddings with
        args (ArgumentParser): script arguments
    Returns:
        (Batch): PyG batch of protein or molecule embedding values.
    """
    embed_failed = False
    data_out = []

    pocket_spec = getattr(input_data_loader, "pocket_spec", None)

    if args.protein and pocket_spec is None:
        log.warning("No pocket specified for %s", input_data_loader.source_file)
        embed_failed = True

    if args.protein and not embed_failed:
        # ProteinEncoder derives the pocket from these coordinates, the same way it does
        # during training and in decoy_eval.
        pocket_coords = Data(pos=torch.as_tensor(pocket_spec).float())
        pocket_coords.batch = torch.zeros(len(pocket_coords.pos), dtype=torch.long)
        pocket_coords = pocket_coords.to(args.device)

    for batch in input_data_loader:
        if embed_failed:
            break

        if args.protein and getattr(batch, "num_graphs", 1) != 1:
            # one pocket spec cannot be shared across several protein graphs
            log.error(
                "%s holds %s protein graphs; embed them one at a time.",
                input_data_loader.source_file,
                batch.num_graphs,
            )
            embed_failed = True
            break

        with torch.no_grad():
            try:
                if args.protein:
                    embeds = encoder(batch.to(args.device), pocket_coords)
                else:
                    embeds = encoder(batch.to(args.device))
            except Exception as e:
                log.error("Error during embedding: %s", e)
                traceback.print_exc()
                embed_failed = True
                break

        if args.molecule:
            batch.x = embeds
            data_out += batch.cpu().to_data_list()

        if args.protein:
            data_out.append(embeds.cpu())

    if embed_failed:
        if args.no_overwrite:
            os.remove(outfile)
        return None

    embeds_out = Batch.from_data_list(data_out)
    return embeds_out


def main(args):
    encoder = get_encoder('p' if args.protein else 'm', args.weights_file, args.device)

    input_list = get_input_list(args)

    output_path = args.output_path

    if output_path[-1] != "/":
        output_path += "/"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for input_line in input_list:
        structure_file = input_line[0]
        structure_file_basename = '.'.join('_'.join(structure_file.split('/')[-args.name_depth:]).split('.')[:-1])

        outfile = (
            output_path + structure_file_basename + "_embeds" + args.suffix + ".pyg"
        )

        if args.no_overwrite:
            # creates an empty file so parallel jobs know to skip current target
            # returns False if file exists
            if handle_no_overwrite(outfile) is False:
                continue

        input_data_loader = get_input_data_loader(input_line, args)
        embeds_out = evaluate_data(input_data_loader, outfile, encoder, args)

        if embeds_out is None:
            continue

        embeds_out.source = structure_file

        torch.save(embeds_out, outfile)
        log.info("completed: %s", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce atom embeddings for protein or small molecule structure"
    )
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
