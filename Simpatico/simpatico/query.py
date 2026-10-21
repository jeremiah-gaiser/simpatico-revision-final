import argparse
import logging
import os
import pickle
import sys
import time
from glob import glob
from os import path
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import faiss
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius

from simpatico import config
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.utils.app_utils import get_encoder
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
    concatenate_pyg_files,
    report_results,
)
from simpatico.utils.faiss_utils import VectorDatabase
from simpatico.utils.mol_utils import get_xyz_from_file, molfile2pyg
from simpatico.utils.pdb_utils import pdb2pyg


def add_arguments(parser):
    parser.add_argument("input_file", type=str, help="R|Path to input file.\n")
    parser.add_argument("output_dir", type=str, help="Directory for results")
    parser.add_argument(
        '-d',
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument("-o", "--output-file")
    parser.add_argument('-w', '--weights',
                        type=str,
                        default=config["default_weights_path"],
                        help='Path to the weights file')

    parser.add_argument('--save-thresholds',
                        type=str,
                        default=None)

    parser.add_argument('--load-thresholds',
                        type=str,
                        default=None)

    parser.add_argument('--results-file', type=str,
                        help='The path for the singular results file (required if one-db is True).')

    parser.set_defaults(main=main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)

def is_binary(tensor: torch.Tensor) -> bool:
    return torch.all((tensor == 0) | (tensor == 1)).item()

def main(args):
    torch.set_grad_enabled(False)
    log_config = {'level': logging.INFO,
                  'format': '%(asctime)s - %(levelname)s - %(message)s',
                  'force': True}
    if args.output_file:
        log_config['filename'] = args.output_file
    else:
        log_config['stream'] = sys.stdout

    logging.basicConfig(**log_config)
    log = logging.getLogger(__name__)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir[-1] != '/':
        args.output_dir += '/'

    query_files = []
    coord_files = []
    db_files = []

    with open(args.input_file) as spec_in:
        for line in spec_in:
            row_data = [x.strip() for x in line.split(",")]

            if len(row_data) == 3:
                data_type, graph_file, coord_file = row_data
                coord_files.append(coord_file)
            else:
                data_type, graph_file = row_data

            data_type = data_type.lower()

            if data_type == "q":
                query_files.append(graph_file)

            if data_type == "d":
                db_files.append(graph_file)

    query_graph_list = []
    pocket_coord_list = []

    for pocket_file, coord_file in zip(query_files, coord_files):
        if pocket_file[-3:] != 'pyg':
            query_g = pdb2pyg(pocket_file)
            query_graph_list.append(query_g)

        pocket_coordinates = get_xyz_from_file(coord_file)
        pocket_coord_list.append(Data(pos=pocket_coordinates))

    query_batch = Batch.from_data_list(query_graph_list)
    coord_batch = Batch.from_data_list(pocket_coord_list)

    encode_query = is_binary(query_batch.x[:100])

    if encode_query:
        query_encoder = get_encoder('p', args.weights, args.device)
        query_encoder.eval()

        with torch.no_grad():
            query_embeds = query_encoder(query_batch.to(args.device), coord_batch.to(args.device))
    else:
        query_embeds = query_batch

    if args.load_thresholds:
        score_thresholds = pickle.load(open(args.load_thresholds, 'rb'))
    else:
        score_thresholds = None

    queries = VectorDatabase(query_embeds, score_thresholds)
    db_encoder = None

    for db_file in db_files:
        log.info(f'Starting {db_file}')
        db_filename = '.'.join(db_file.split('/')[-1].split('.')[:-1])
        output_filename = db_filename + '_query-results.csv'

        db_batch, _ = molfile2pyg(db_file)
        encode_db = is_binary(db_batch.x[:100])

        if encode_db:
            if db_encoder is None:
                db_encoder = get_encoder('m', args.weights, args.device)

            log.info(f'Start encoding')
            db_outs = []
            db_data_loader = DataLoader(db_batch, batch_size=1024, shuffle=False)

            for batch in db_data_loader:
                with torch.no_grad():
                    embeds = db_encoder(batch.to(args.device))
                batch.x = embeds
                db_outs += batch.cpu().to_data_list()

            db_embeds = Batch.from_data_list(db_outs)
            log.info(f'Encoding complete')
        else:
            db_embeds = db_batch

        vector_db = VectorDatabase(db_embeds)

        if queries.score_thresholds is None or args.save_thresholds:
            queries.get_score_thresholds(vector_db)
            if args.save_thresholds:
                with open(args.save_thresholds, 'wb') as thresholds_out:
                    pickle.dump(queries.score_thresholds, thresholds_out)

                sys.exit(f'Saved query scoring thresholds in {args.save_thresholds}.')

        search_results = vector_db.query(queries)
        output_csv_content = ''

        for target_idx in range(len(search_results)):
            for m_score, m_idx in zip(*search_results[target_idx]):
                output_csv_content += f"{target_idx+1},{m_idx+1},{m_score}\n"

        with open(args.output_dir + output_filename, 'w') as results_out:
            results_out.write(output_csv_content)

        log.info(f"Successfully completed {db_file}")
