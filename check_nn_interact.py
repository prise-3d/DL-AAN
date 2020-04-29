import os
import numpy as np
import time

from PIL import Image


default_folder = 'data'
input_folder = os.path.join(default_folder, 'input')
output_folder = os.path.join(default_folder, 'output')

def output():

    files = sorted(os.listdir(output_folder))

    filtered_files = [ os.path.join(output_folder, f) for f in files ]

    arrays = []
    temp_array = []

    counter = 1

    for f in filtered_files:
        a = np.load(f)
        
        temp_array.append(a)

        # new line (32 x 1920 x 3)
        if counter % 60 == 0:

            if a.ndim > 2:
                arrays.append(np.concatenate(temp_array, axis=2).transpose((2, 1, 0)))
            else:
                arrays.append(np.concatenate(temp_array, axis=1))

            temp_array = []
        
        counter += 1

    concatenated = np.array(arrays)

    concatenated = np.concatenate(concatenated.transpose((0, 2, 1, 3)), axis=0)

    
    if concatenated.ndim > 2:
        img = concatenated.reshape((1088, 1920, 3))
    else:
        img = concatenated.reshape((1088, 1920))
    
    img = np.array(img * 255, 'uint8')
    Image.fromarray(img).show()


def input(to_read):

    files = sorted(os.listdir(input_folder))

    filtered_files = [ os.path.join(input_folder, f) for f in files if to_read in f ]

    arrays = []
    temp_array = []

    counter = 1

    for f in filtered_files:
        a = np.load(f)
        
        temp_array.append(a)

        # new line (32 x 1920 x 3)
        if counter % 60 == 0:

            if a.ndim > 2:
                arrays.append(np.concatenate(temp_array, axis=2).transpose((2, 1, 0)))
            else:
                arrays.append(np.concatenate(temp_array, axis=1))

            temp_array = []
        
        counter += 1

    concatenated = np.array(arrays)
    if to_read != 'zbuffer':
        concatenated = np.concatenate(concatenated.transpose((0, 2, 1, 3)), axis=0)
    else:
        concatenated = np.concatenate(concatenated, axis=0)

    
    if concatenated.ndim > 2:
        img = concatenated.reshape((1088, 1920, 3))
    else:
        img = concatenated.reshape((1088, 1920))
    
    img = np.array(img * 255, 'uint8')
    Image.fromarray(img).show()

if __name__ == "__main__":
    input('spectrum')
    input('normals')
    input('zbuffer')
    output()