# main imports
import numpy as np
import argparse
import os
import math
import sys

# image processing
from PIL import Image

# deep learning imports
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# vizualisation
import torchvision.utils as vutils
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# models imports
from models.autoencoders.ushaped_dropout import UShapedAutoencoder as AutoEncoder
from models.discriminators.discriminator_v1 import Discriminator

# losses imports
from losses.utils import instanciate, loss_choices

# logger import
import gym
log = gym.logger
log.set_level(gym.logger.INFO)

# other parameters
NB_IMAGES = 64

BACKUP_MODEL_NAME = "{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0002
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 20

'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")


def main():

    parser = argparse.ArgumentParser(description="Denoise folder of image using Autoencoder")

    parser.add_argument('--features', type=str, help="images input path (ex: \"image1.png image2.png image3.png\")")
    parser.add_argument('--tile_size', type=str, help='specify size of the tile used', default='32,32')
    parser.add_argument('--load', type=str, help='folder backup model', default='')
    parser.add_argument('--output', type=str, help='output reconstructed image', default='reconstructed.png')

    args = parser.parse_args()

    p_features  = args.features.split(' ')
    p_tile      = args.tile_size.split(',')
    p_load      = args.load
    p_output    = args.output

    tile_size = int(p_tile[0]), int(p_tile[1])

    img_array_list = []

    # prepare data
    for img_path in p_features:
        
        img_array = np.array(Image.open(img_path))


        print(img_path)
        print('min', np.min(img_array))
        print('mean', np.mean(img_array))
        print('max', np.max(img_array))

        img_array = img_array / 255.
        
        if img_array.ndim < 3:
        
            # add dimensions
            img_array = np.expand_dims(img_array, axis=2)
        
        img_array = img_array.transpose(2, 0, 1)
        img_array_list.append(img_array)

    input_data = np.concatenate(img_array_list, axis=0)

    print('Input features model shape', input_data.shape)

    c, h, w = input_data.shape

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # define models and loss functions
    autoencoder = AutoEncoder(c, tile_size[0])
    #.to(device)

     # prepare folder names to save models
    load_models_folder_path = os.path.join(BACKUP_FOLDER, p_load)

    load_autoencoder_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    # load autoencoder state
    autoencoder_checkpoint = torch.load(load_autoencoder_model_path, map_location=device)

    autoencoder.load_state_dict(autoencoder_checkpoint['autoencoder_state_dict'])
    autoencoder.eval()

    # get tiles from the image
    output_array = np.ones((3, h, w))
    print('Output shape', output_array.shape)


    n_h_tiles = math.ceil(h / tile_size[0])
    n_w_tiles = math.ceil(w / tile_size[1]) 

    number_of_calls = n_h_tiles * n_w_tiles
    image_counter = 0

    for i in range(n_h_tiles):
        for j in range(n_w_tiles):

            h_start = i * tile_size[0]
            w_start = j * tile_size[1]

            # classical end
            h_end = h_start + tile_size[0]
            w_end = w_start + tile_size[1]

            # check and avoid out of bounds
            if h_end >= h and w_end >= w:
                h_final_start = h - tile_size[0]
                w_final_start = w - tile_size[1]

            elif h_end >= h:
                h_final_start = h - tile_size[0]
                w_final_start = w_start

            elif w_end >= w:
                h_final_start = h_start
                w_final_start = w - tile_size[1]
             
            else:
                h_final_start = h_start
                w_final_start = w_start
            
            # final start and end
            h_end = h_final_start + tile_size[0]
            w_end = w_final_start + tile_size[1]

            # extract data
            input_tile = input_data[:, h_final_start:h_end, w_final_start:w_end]
            input_tile = np.expand_dims(input_tile, axis=0)
            # input_tile = np.array(input_tile / 255.)
            
            #print('model input', input_tile.shape)

            # predict output_data
            output_tile = autoencoder(torch.from_numpy(input_tile).float())

            # replace using h_final and w_final
            # print('(', h_final_start, '-', h_end, ') | (', w_final_start, '-', w_end, ')')

            np_output_tile = np.squeeze(output_tile.detach().numpy())
            
            print('min', np.min(np_output_tile))
            print('mean', np.mean(np_output_tile))
            print('max', np.max(np_output_tile))

            output_array[:, h_final_start:h_end, w_final_start:w_end] = np_output_tile

            write_progress((image_counter + 1) / number_of_calls)
            image_counter += 1

    # update color values 
    output_array = output_array * 255
    # retranspose output data
    output_array = output_array.transpose(1, 2, 0)

    Image.fromarray(np.array(output_array, 'uint8')).save(p_output)

if __name__ == "__main__":
    main()