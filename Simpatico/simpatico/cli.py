import argparse
from simpatico import train, evaluate, convert, get_train_set, query, print_results, get_biolip_samples, decoy_eval


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for Simpatico: train models, evaluate predictions, convert structures, prepare datasets, and query embedding databases.",
    )

    # first argument defines which parser to use.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # instantiate a new subparser
    eval_parser = subparsers.add_parser(
        "eval",
        help="Produce atom embeddings for protein pockets or small molecules.",
        description="Produce atom embeddings for protein pockets or small molecules.\n"
        "input: descriptive batch or individual structure file (.csv, .pdb, .smi, .ism, .sdf, .pdb)\n"
        "output: .pyg file of atomic embeddings",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # build up subparser from specifications in evaluate.py and sets parser's 'main' function to evaluate.main
    evaluate.add_arguments(eval_parser)

    # do the same for the other methods command line accessible methods below
    train_parser = subparsers.add_parser("train", help="Train or update model weights")
    train.add_arguments(train_parser)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert protein or small molecule structures to PyG graphs"
    )
    convert.add_arguments(convert_parser)

    get_train_set_parser = subparsers.add_parser(
        "get-train-set",
        help="Convert protein and molecule files into PyG graphs and consolidate into a train-validation set.",
    )
    get_train_set.add_arguments(get_train_set_parser)

    query_parser = subparsers.add_parser(
        "query",
        help="Query a database of simpatico embeddings of with a different set of simpatico embeddings (virtual screening procedure).",
    )
    query.add_arguments(query_parser)

    print_results_parser = subparsers.add_parser(
        "print-results", help="Print results from a query/virtual screen."
    )
    print_results.add_arguments(print_results_parser)

    get_biolip_parser = subparsers.add_parser(
        'get-biolip-samples', help='Generate training graph samples from the biolip database index and corresponding .pdb files.'
    )
    get_biolip_samples.add_arguments(get_biolip_parser)

    decoy_eval_parser = subparsers.add_parser(
        'decoy-eval', help='Evaluate trained model against decoy dataset.'
    )
    decoy_eval.add_arguments(decoy_eval_parser)


    args = parser.parse_args()

    # invoke the main function from specified metho
    args.main(args)
