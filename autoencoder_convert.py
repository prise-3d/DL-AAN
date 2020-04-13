# main imports
import os
import argparse

from models.autoencoders.ushaped import UShapedAutoencoder as Autoencoder

# dl imports
import torch
import torchvision

# variables
BACKUP_FOLDER = "saved_models"
BACKUP_MODEL_NAME = "{}_model.pt"

def main():

    parser = argparse.ArgumentParser(description="Generate model using specific features")

    parser.add_argument('--output', type=str, help="output file expected (must be .pt extension)", required=True)
    parser.add_argument('--channels', type=int, help="Number of expected channels", required=True)
    parser.add_argument('--tile_size', type=str, help='specify size of the tile used', default='32,32')
    parser.add_argument('--load', type=str, help='folder backup model', required=True)

    args = parser.parse_args()

    p_channels            = args.channels
    p_tile                = args.tile_size.split(',')
    p_load                = args.load
    p_output              = args.output

    tile_size = int(p_tile[0]), int(p_tile[1])

    # An instance of your model.
    autoencoder = Autoencoder(p_channels, tile_size)
    load_models_folder_path = os.path.join(BACKUP_FOLDER, p_load)
    load_autoencoder_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    # as PBRT works only on CPU, let by default CPU
    autoencoder_checkpoint = torch.load(load_autoencoder_model_path, map_location='cpu')

    autoencoder.load_state_dict(autoencoder_checkpoint['autoencoder_state_dict'])
    autoencoder.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 7, 32, 32)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(autoencoder, example)

    if '/' in p_output:
        head, _ = os.path.split(p_output)

        if not os.path.exists(head):
            os.makedirs(head)

    traced_script_module.save(p_output)

if __name__ == "__main__":
    main()