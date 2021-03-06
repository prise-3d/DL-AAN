# main imports
import sys
import os
import subprocess
import struct
import time
import numpy

# ml imports
import torch
from torch.autograd.variable import Variable

# models imports
from models.autoencoders.ushaped import UShapedAutoencoder as AutoEncoder
from models.discriminators.discriminator_v1 import Discriminator

nb_call_nn = 0

# -----------------------------------------------------------------------------
# Initialize current root dir

pydir = os.path.dirname(os.path.abspath(__file__)) # root/ml
rootdir = os.path.dirname(pydir)
os.chdir(rootdir)

# Constants
IMAGE_TILE_SIZE = 32
INPUT_N_CHANNELS = 7

BACKUP_FOLDER = "ml/saved_models"
BACKUP_MODEL_NAME = "{}_model.pt"

DEFAULT_MODEL = 'aan1_samples_based_norm'
DEFAULT_MODEL_PATH = os.path.join(rootdir, BACKUP_FOLDER, DEFAULT_MODEL)

# =============================================================================
# Utilities

def print_stderr(s):
    sys.stderr.write(s + "\n")

def write_char(c):
    sys.stdout.buffer.write(c.encode())

# <return> a numpy 1D array
def read_float_array(num):
    buff = sys.stdin.buffer.read(num * 4)
    return numpy.frombuffer(buff, dtype=numpy.float32)

# <return> a (7, height, width) shaped ndarray
def read_input():
    # Read rgb spectrum data
    spectrumArray = read_float_array(IMAGE_TILE_SIZE * IMAGE_TILE_SIZE * 3)

    # Read normals data
    normalsArray = read_float_array(IMAGE_TILE_SIZE * IMAGE_TILE_SIZE * 3)

    # Read distance data
    distanceArray = read_float_array(IMAGE_TILE_SIZE * IMAGE_TILE_SIZE * 1)

    # Reshape read arrays into (height, width, channels)
    spectrumArray = spectrumArray.reshape((3, IMAGE_TILE_SIZE, IMAGE_TILE_SIZE))
    normalsArray = normalsArray.reshape((3, IMAGE_TILE_SIZE, IMAGE_TILE_SIZE))
    distanceArray = distanceArray.reshape((1, IMAGE_TILE_SIZE, IMAGE_TILE_SIZE))

    # Concatenate into single multiarray
    return numpy.concatenate([spectrumArray, normalsArray, distanceArray], axis=0)

# =============================================================================
# <nparray> a shape (channel, height, width) 3D ndarray
# Outputted as an image with dimensions order as (height, width, channel)
def output_to_stdout(nparray):
    # Reshape into (height, width, channel)
    # data = numpy.transpose(nparray, (1, 2, 0))
    # buff = data.tobytes()
    buff = nparray.tobytes()
    sys.stdout.buffer.write(buff)
    write_char("x")
    write_char("\n")
    sys.stdout.flush()

# =============================================================================
# Processing function
def process_one(net):

    # Read input from stdin
    inputNdArray = read_input()

    global nb_call_nn

    index_str = str(nb_call_nn)

    while len(index_str) < 5:
        index_str = "0" + index_str

    spectrum = inputNdArray[0:3, :, :]
    normals = inputNdArray[3:6, :, :]
    zbuffer = inputNdArray[6, :, :]

    if not os.path.exists('data/input'):
        os.makedirs('data/input')

    numpy.save('data/input/spectrum_' + index_str, spectrum)
    numpy.save('data/input/normals_' + index_str, normals)
    numpy.save('data/input/zbuffer_' + index_str, zbuffer)

    torchData = torch.from_numpy(inputNdArray).float()
    torchData = torchData.unsqueeze(0)
    inputVariable = Variable(torchData)

    # Run the network
    outputVariable = net(inputVariable)

    outputNdArray = outputVariable.data.numpy()[0]
    output_to_stdout(outputNdArray)

    if not os.path.exists('data/output'):
        os.makedirs('data/output')

    numpy.save('data/output/output_' + index_str, outputNdArray)

    nb_call_nn += 1

# =============================================================================
# Main

def main():

    # print_stderr("nn_interact.py: Startup")
    torch.set_num_threads(1)
    
    # Load model
    net = AutoEncoder(INPUT_N_CHANNELS, IMAGE_TILE_SIZE)

    # get state dict of model
    load_autoencoder_model_path = os.path.join(DEFAULT_MODEL_PATH, BACKUP_MODEL_NAME.format('autoencoder'))
    autoencoder_checkpoint = torch.load(load_autoencoder_model_path, map_location='cpu')

    net.load_state_dict(autoencoder_checkpoint['autoencoder_state_dict'])

    # Put in eval mode
    net.eval()

    # print_stderr("Model loaded")

    while True:
        process_one(net)

main()