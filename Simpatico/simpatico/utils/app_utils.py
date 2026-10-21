import torch
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder

def get_encoder(encoder_type, weight_file, device='cpu'):
    """
    Get encoder according to input type (either protein or molecular structural file, or batch file).

    Args:
        args (ArgumentParser): script arguments
    Returns:
        encoder  (MolEncoder | ProteinEncoder): proper encoder module
    """
    # weights for protein and ligand models are stored in one file.
    # first item is protein model weights, second is molecule model weights
    # so set `weight_index` accordingly
    if encoder_type=='p':
        encoder = ProteinEncoder().to(device)
        weight_index = 0

    elif encoder_type=='m':
        encoder = MolEncoder().to(device)
        weight_index = 1

    encoder.load_state_dict(
        torch.load(weight_file, map_location=device)[weight_index]
    )
    encoder.eval()

    return encoder