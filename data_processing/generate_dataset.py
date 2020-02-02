# main imports
import argparse
import numpy as np
import os, sys

# image imports
from PIL import Image

# others imports
import math
import random

# important variables
data_train_folder  = 'train'
data_test_folder   = 'test'

data_noises_folder = 'noises'
data_ref_folder    = 'references'

tile_size          = (32, 32)

number_of_images   = 0 # used for writing extraction progress 
images_counter     = 0 # counter used for extraction progress


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


'''
Constuct all dataset with specific tile size
'''
def construct_tiles(scenes, noises_path, references_path, output_path, nb):

    global images_counter

    h_tile, w_tile = tile_size

    ref_scene_output_path = os.path.join(output_path, data_ref_folder)
    noises_scene_output_path = os.path.join(output_path, data_noises_folder)

    if not os.path.exists(ref_scene_output_path):
        os.makedirs(ref_scene_output_path)
        os.makedirs(noises_scene_output_path)

    output_img_index = 0

    for scene in scenes:

        # get noises images name
        scene_path = os.path.join(noises_path, scene)
        noises_images = os.listdir(scene_path)

        # get reference image path 
        reference_image_name = os.listdir(os.path.join(references_path, scene))[0]
        reference_image_path = os.path.join(references_path, scene, reference_image_name)
        reference_image = np.array(Image.open(reference_image_path))

        # create tile for each image
        for img_name in noises_images:

            images_counter = images_counter + 1

            img_path = os.path.join(scene_path, img_name)
            img = np.array(Image.open(img_path))

            h, w, _ = img.shape

            for _ in range(nb):

                output_img_index = output_img_index + 1

                h_random = random.randint(0, h - h_tile - 1)
                w_random = random.randint(0, w - w_tile - 1)

                h_end = h_random+h_tile
                w_end = w_random+w_tile
                tile_extract_noisy = img[h_random:h_end, w_random:w_end]
                tile_extract_ref = reference_image[h_random:h_end, w_random:w_end]
                
                output_index_str = str(output_img_index)

                while len(output_index_str) < 11:
                    output_index_str = '0' + output_index_str

                output_image_name = output_index_str + '_' + scene + '.png'

                output_noisy_path = os.path.join(noises_scene_output_path, output_image_name)
                output_ref_path = os.path.join(ref_scene_output_path, output_image_name)

                Image.fromarray(np.array(tile_extract_noisy, 'uint8')).save(output_noisy_path)
                Image.fromarray(np.array(tile_extract_ref, 'uint8')).save(output_ref_path)


            # write progress using global variable
            write_progress((images_counter + 1) / number_of_images)


def main():

    global number_of_images, images_counter

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--noises', type=str, help="folder scenes with noises data")
    parser.add_argument('--references', type=str, help='folder scenes with references')
    parser.add_argument('--nb', type=int, help='number of tile extracted from each images')
    parser.add_argument('--output', type=str, help='output folder of whole data `test` and `train` folder')
    parser.add_argument('--train_split', type=float, help='test split size of generated data (based of number of scenes)', default=0.2)

    args = parser.parse_args()

    p_noises     = args.noises
    p_references = args.references
    p_nb         = args.nb
    p_output     = args.output
    p_split      = args.train_split


    # get list scenes folders and shuffle it
    scenes_folder = os.listdir(p_noises)
    random.shuffle(scenes_folder)

    nb_scenes = len(scenes_folder)
    nb_scenes_test = math.ceil(nb_scenes * p_split)

    test_scenes = scenes_folder[:nb_scenes_test]
    train_scenes = scenes_folder[nb_scenes_test:]

    print('------------------------------------------------------------------------------------------------------')
    print('Train scenes :', train_scenes)
    print('------------------------------------------------------------------------------------------------------')
    print('Test scenes :', test_scenes)

    scenes_path = [ os.path.join(p_noises, scene) for scene in scenes_folder ]
    number_of_images = sum([ len(os.listdir(scene)) for scene in scenes_path ])

    print('------------------------------------------------------------------------------------------------------')
    print('-- Start generating data')
    print('------------------------------------------------------------------------------------------------------')

    output_test_folder = os.path.join(p_output, data_test_folder)
    output_train_folder = os.path.join(p_output, data_train_folder)

    # contruct test tiles
    construct_tiles(test_scenes, p_noises, p_references, output_test_folder, p_nb)

    # construct train tiles
    construct_tiles(train_scenes, p_noises, p_references, output_train_folder, p_nb)

    # write progress using global variable
    write_progress((images_counter + 1) / number_of_images)
    

if __name__ == "__main__":
    main()