import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from simpatico.models import load_encoders
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
    # Required: the architecture is read from the arch.json beside the weights (load_encoders),
    # and the configured default checkpoint is not shipped.
    parser.add_argument('-w', '--weights',
                        type=str,
                        required=True,
                        help='Path to the weights file (arch.json must sit beside it)')

    parser.add_argument('--save-thresholds',
                        type=str,
                        default=None,
                        help='Compute per-query thresholds against the FIRST database file, '
                             'save them here, and exit.')

    parser.add_argument('--load-thresholds',
                        type=str,
                        default=None,
                        help='Score every database file against thresholds saved by --save-thresholds.')

    parser.add_argument('--quantile', type=float, default=0.99,
                        help='Background score quantile for the per-voxel threshold (default 0.99).')

    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip database files whose results .pkl already exists (for resuming). '
                             'Requires --load-thresholds so resumed files share thresholds.')

    parser.set_defaults(main=main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)

def is_binary(tensor: torch.Tensor) -> bool:
    return torch.all((tensor == 0) | (tensor == 1)).item()

def load_graph(graph_file):
    """Receptor or molecule file -> PyG graph. molfile2pyg returns (batch, smiles) for structure
    files but a bare batch for .pyg, so unify here."""
    if graph_file.endswith('.pyg'):
        return torch.load(graph_file, weights_only=False)
    return molfile2pyg(graph_file)[0]

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

    if args.skip_existing and not args.load_thresholds:
        sys.exit('--skip-existing requires --load-thresholds.')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir[-1] != '/':
        args.output_dir += '/'

    query_files = []
    coord_files = []
    db_files = []

    with open(args.input_file) as spec_in:
        for line in spec_in:
            if not line.strip():
                continue

            row_data = [x.strip() for x in line.split(",")]
            data_type = row_data[0].lower()

            if data_type == "q":
                if len(row_data) != 3:
                    sys.exit(f'query row needs a pocket-coordinate file: {line.strip()}')
                query_files.append(row_data[1])
                coord_files.append(row_data[2])

            if data_type == "d":
                db_files.append(row_data[1])

    # load_encoders reads arch.json beside the weights: PocketEncoder for the "pocket" kind, and
    # MolEncoder with the trained max_hops. get_encoder() ignored both, building the voxel
    # ProteinEncoder (whose state dict does not fit these checkpoints) and max_hops=3.
    protein_encoder, mol_encoder = load_encoders(args.weights, device=args.device)
    log.info(f'Loaded {args.weights} ({type(protein_encoder).__name__}, '
             f'max_hops={mol_encoder.max_hops})')

    query_graph_list = []
    pocket_coord_list = []

    for pocket_file, coord_file in zip(query_files, coord_files):
        query_g = pdb2pyg(pocket_file) if pocket_file[-3:] != 'pyg' else load_graph(pocket_file)
        query_graph_list.append(Data(x=query_g.x, pos=query_g.pos.float()))
        pocket_coord_list.append(Data(pos=get_xyz_from_file(coord_file).float()))

    query_batch = Batch.from_data_list(query_graph_list)
    coord_batch = Batch.from_data_list(pocket_coord_list)

    if is_binary(query_batch.x[:100]):
        query_embeds = protein_encoder(query_batch.to(args.device), coord_batch.to(args.device))
    else:
        query_embeds = query_batch

    voxel_counts = torch.bincount(query_embeds.batch, minlength=len(query_files)).tolist()
    log.info(f'{len(query_files)} queries, {sum(voxel_counts)} pocket voxels '
             f'(min {min(voxel_counts)}, max {max(voxel_counts)} per query)')

    if args.load_thresholds:
        score_thresholds = pickle.load(open(args.load_thresholds, 'rb'))
        if len(score_thresholds[0]) != len(query_files):
            sys.exit(f'{args.load_thresholds} holds thresholds for {len(score_thresholds[0])} '
                     f'queries, but {args.input_file} lists {len(query_files)}.')
        log.info(f'Loaded score thresholds from {args.load_thresholds}')
    else:
        score_thresholds = None

    queries = VectorDatabase(query_embeds, score_thresholds)

    for db_file in db_files:
        db_filename = '.'.join(db_file.split('/')[-1].split('.')[:-1])
        output_path = args.output_dir + db_filename + '.pkl'

        if args.skip_existing and os.path.exists(output_path):
            log.info(f'Skipping {db_file}: {output_path} exists')
            continue

        log.info(f'Starting {db_file}')
        db_batch = load_graph(db_file)

        if is_binary(db_batch.x[:100]):
            log.info(f'Start encoding')
            db_outs = []
            db_data_loader = DataLoader(db_batch, batch_size=1024, shuffle=False)

            for batch in db_data_loader:
                embeds = mol_encoder(batch.to(args.device))
                batch.x = embeds
                db_outs += batch.cpu().to_data_list()

            db_embeds = Batch.from_data_list(db_outs)
            log.info(f'Encoding complete')
        else:
            db_embeds = db_batch

        vector_db = VectorDatabase(db_embeds)

        if queries.score_thresholds is None or args.save_thresholds:
            queries.get_score_thresholds(vector_db, q=args.quantile)
            log.info(f'Computed score thresholds against {db_file} (q={args.quantile})')
            if args.save_thresholds:
                with open(args.save_thresholds, 'wb') as thresholds_out:
                    pickle.dump(tuple(x.cpu() for x in queries.score_thresholds), thresholds_out)

                log.info(f'Saved query scoring thresholds in {args.save_thresholds}.')
                return

        search_results = vector_db.query(queries)

        # Same format as earlier screens: per query, (sorted non-zero scores, molecule indices
        # within this file). Written to a temporary name first so a job killed mid-write never
        # leaves a truncated .pkl for --skip-existing to trust.
        with open(output_path + '.tmp', 'wb') as results_out:
            pickle.dump(search_results, results_out)
        os.replace(output_path + '.tmp', output_path)

        log.info(f"Successfully completed {db_file}")
