import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import sys
from glob import glob
from os import path
from simpatico import config
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico.utils.mol_utils import molfile2pyg, get_xyz_from_file
from filelock import FileLock, Timeout



def add_arguments(parser):
    parser.add_argument(
        "input",
        help='List of file paths or quote-bound unix-style path (e.g. "/path/to/data/*.pdb") describing input data.',
    )
    parser.add_argument("output_directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--protein",
        action="store_true",
        help="convert input files to protein graphs",
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="convert input files to small molecule structures",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="optional suffix to insert between filename and extension in converted PyG files (file<suffix>.pyg)",
    )
    parser.add_argument(
        '--skip-smiles',
        action="store_true",
        help='do not save a new .ism file alongside batch of molecule graphs'
    )
    parser.add_argument(
        '--name-depth',
        type=int,
        default=1,
        help='parent directory depth used to generate new filename. e.g. for name_depth=2, /path/to/file.sdf -> to_file.pyg'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true'
    )

    # sets parser's main function to the main function in this script
    parser.set_defaults(main=main)


def gather_structure_files(input_string) -> list[str]:
    """
    Parse input string and generate list of input structural files.
    Args:
        input_string (str): Path to input, either batch file (.txt, .csv)
    Returns:
        (list[str]): list of structure files to convert to PyG graphs.
    """
    _, input_filetype = path.splitext(input_string)
    # filetypes we directly convert
    structure_filetypes = config["protein_filetypes"] + config["molecule_filetypes"]

    # if input is a list, we need to process the file.
    # otherwise, gather files with glob function
    input_is_list = input_filetype not in structure_filetypes

    if input_is_list:
        structure_files = []
        with open(input_string, "r") as structure_file_list:
            # filter out any length 0 lines from structure file list
            for line in structure_file_list:
                line = line.strip()
                
                if len(line) == 0:
                    continue

                structure_files.append([x.strip() for x in line.split(',')])

        return structure_files
    else:
        return [[x] for x in glob(input_string)]


def new_filename(input_file, extension, output_dir, suffix=None, name_depth=1):
    """
    Generate the the filename to be used for file converted to PyG.
    Args:
        input_file (str): path to input filename.
        extension (str): file extension to use (usually .pyg or .pkl).
        output_dir (str): directory where output file will be stored.
        suffix (str, optional): string to append to file name before extension.
        name_depth (int): parent directory depth to prepend to new filename, e.g. for name_depth=2, /path/to/file.sdf -> to_file.pyg
    Returns:
        (str): path to new PyG file.
    """
    path_list = input_file.split('/')
    basename = '.'.join('_'.join(path_list[-name_depth:]).split('.')[:-1])
    output_dir = output_dir + "/" if output_dir[-1] != "/" else output_dir
    extension = "." + extension if extension[0] != "." else extension

    if suffix is None:
        suffix = ""

    new_path = f"{output_dir}{basename}{suffix}{extension}"
    return new_path


def main(args):
    structure_files = gather_structure_files(args.input)
    converter = pdb2pyg if args.protein else molfile2pyg

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    for sf_row in tqdm(structure_files, desc='Converting structures:'):
        sf = sf_row[0]
        ligand_coords = None

        pyg_file_out = new_filename(sf, ".pyg", args.output_directory, args.suffix, name_depth=args.name_depth)

        if args.no_overwrite:
            if path.exists(pyg_file_out):
                continue
        
        lockfile = pyg_file_out + '.lock'

        try:
            with FileLock(lockfile, timeout=1):
                if len(sf_row) > 1:
                    mol_file = sf_row[1]
                    ligand_coords = get_xyz_from_file(mol_file)

                if args.protein:
                    if ligand_coords is not None:
                        pyg_graph = converter(sf, ligand_pos=ligand_coords, pocket_coords=ligand_coords)
                    else:
                        pyg_graph = converter(sf)

                    torch.save(pyg_graph, pyg_file_out)
                else:
                    pyg_graph, smiles = converter(sf)
                    torch.save(pyg_graph, pyg_file_out)

                    if args.skip_smiles == False:
                        smile_file_out = pyg_file_out.replace('.pyg', '_out.ism')
                        smile_file_content = '\n'.join(smiles) + '\n'
                        pyg_graph.source = smile_file_out

                        with open(smile_file_out, 'w') as smile_out:
                            smile_out.write(smile_file_content)
        except Timeout:
            continue
