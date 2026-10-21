# scripts/train.py
import argparse
import logging
import os
import pickle

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from simpatico import config
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.query import VectorDatabase
from simpatico.utils.mol_utils import get_xyz_from_file, molfile2pyg
from simpatico.utils.pdb_utils import pdb2pyg


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "input_file",
    )
    parser.add_argument(
        "-w",
        "--weights-file",
        default=config["default_weights_path"],
        help="Non-default weights",
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Just generate pyg graphs for future evaluations, don't actually perform evaluation.",
    )
    parser.add_argument('--save-embeds', default=None)
    parser.add_argument(
        '-o',
        '--output-log'
    )
    parser.add_argument('--device')

    parser.set_defaults(main=main)
    return parser

def process_input_file(input_csv):
    file_data = []
    with open(input_csv, 'r') as csv_in:
        for line in csv_in:
            file_data.append([x.strip() for x in line.split(',')])
    return file_data

def graph_version(f, name_depth=1):
    extension = f.split('.')[-1]
    return f.replace(extension, 'pyg')

def get_graph_dict(target_row):
    graph_data = {}

    for f in target_row:
        graph_data[f] = None

    for f in target_row:
        gf = graph_version(f)

        if os.path.exists(gf):
            with open(gf, 'rb') as pkl_in:
                graph_data[f] = pickle.load(pkl_in)

    target_file, ligand_file, actives_file, decoys_file = target_row

    if graph_data[target_file] is None:
        ligand_coords = get_xyz_from_file(ligand_file)
        graph_data[target_file] = pdb2pyg(target_file, pocket_coords=ligand_coords)

    for mol_file in [actives_file, decoys_file]:
        if graph_data[mol_file] is None:
            graph_data[mol_file] = molfile2pyg(mol_file, ignore_pos=True)

    return graph_data

def save_graphs(graph_dict):
    for k,v in graph_dict.items():
        gf = graph_version(k)
        if os.path.exists(gf) == False:
            with open(gf, 'wb') as pkl_out:
                pickle.dump(v, pkl_out)

def evaluate_mols(mol_graph, encoder, device):
    data_out = []

    mol_loader = DataLoader(mol_graph, batch_size=1024, shuffle=False)

    for batch in mol_loader:
        embeds = encoder(batch.to(device))
        batch.x = embeds
        data_out += batch.to_data_list()

    output_batch = Batch.from_data_list(data_out)
    output_batch.source = mol_graph.source
    return output_batch


def calculate_efs(rank_index, num_actives, num_total, sampling_ratios=[0.001, 0.005, 0.01, 0.05]):
    ef_vals = []
    expected_active_ratio = num_actives / num_total

    for sr in sampling_ratios:
        sample_size = int(num_total * sr)
        sample = rank_index[:sample_size]
        true_active_ratio = (sample < num_actives).sum() / sample_size
        ef = true_active_ratio / expected_active_ratio
        ef_vals.append(ef.item())

    return ef_vals

def main(args):
    device = args.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.output_log:
        log_config = {'level': logging.INFO,
                    'format': '%(asctime)s - %(levelname)s - %(message)s',
                    'force': True}

        log_config['filename'] = args.output_log

        logging.basicConfig(**log_config)
        log = logging.getLogger(__name__)
    else:
        log = None

    if args.graphs_only == False:
        protein_encoder = ProteinEncoder().to(device)
        protein_encoder.load_state_dict(
            torch.load(args.weights_file, map_location=device)[0]
        )
        mol_encoder = MolEncoder().to(device)
        mol_encoder.load_state_dict(
            torch.load(args.weights_file, map_location=device)[1]
        )
        protein_encoder.eval()
        mol_encoder.eval()

    all_ef_results = []

    for target_row in process_input_file(args.input_file):
        if log:
            log.info(f"Starting {target_row[0]}")

        graph_data = get_graph_dict(target_row)

        save_graphs(graph_data)

        if args.graphs_only:
            continue

        target_graph, _, actives_graph, decoys_graph = [graph_data[x] for x in target_row]

        target_graph = target_graph.to(device)

        num_actives = actives_graph.batch[-1]+1
        num_total = num_actives + decoys_graph.batch[-1]+1

        with torch.no_grad():
            pocket_coords = Data(pos=get_xyz_from_file(target_row[1]))
            pocket_coords.batch = torch.zeros(len(pocket_coords.pos))

            protein_out = protein_encoder(target_graph, pocket_coords.to(device))
            target_embeddings = protein_out.x

            target_batch = Batch.from_data_list([Data(x=target_embeddings, source=target_graph.source)])

            if log:
                log.info('Starting eval.')
            active_embeddings = evaluate_mols(actives_graph, mol_encoder, device)
            active_embeddings.source = actives_graph.source

            decoy_embeddings = evaluate_mols(decoys_graph, mol_encoder, device)
            decoy_embeddings.source = decoys_graph.source

            all_embeddings = Batch.from_data_list(active_embeddings.to_data_list() +
                                                  decoy_embeddings.to_data_list())

            if args.save_embeds:
                p_embed_f, _, a_embed_f, d_embed_f = [args.save_embeds + '_'.join(x.split('/')[-2:]) for x in target_row]
                for e,f in zip([protein_out, active_embeddings, decoy_embeddings],
                               [p_embed_f, a_embed_f, d_embed_f]):
                    pickle.dump(e, open(f, 'wb'))

            target_db = VectorDatabase(target_batch)
            vector_db = VectorDatabase(all_embeddings)

            if log:
                log.info('Completed eval.')

            target_db.get_score_thresholds(vector_db)

            screen_ranking = vector_db.query(target_db)[0][1]
            ef_vals = calculate_efs(torch.tensor(screen_ranking), num_actives, num_total)
            all_ef_results.append([target_row[0], ef_vals])

            if log:
                log.info('Completed decoy eval.')

            print(target_row[0] + ', ' + ', '.join([f"{x:0.2f}" for x in ef_vals]))
            ef_avgs = torch.tensor([x[1] for x in all_ef_results]).mean(0).tolist()


    if args.graphs_only == False:
        print('EF averages: ' + ', '.join([f"{x:0.2f}" for x in ef_avgs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model against a decoy dataset."
    )
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
