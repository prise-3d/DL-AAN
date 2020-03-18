# main imports
import argparse
import numpy as np
import os, sys

# image imports
from PIL import Image

# others imports
import math
import random

# modules imports
import config as cfg

# important variables
data_train_folder  = 'train'
data_test_folder   = 'test'

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
def construct_tiles(scenes, main_path, features, statics_path, references_path, output_path, nb):

    global images_counter

    h_tile, w_tile = tile_size

    # compute features path
    features_path = [ os.path.join(main_path, feature) for feature in features ]

    # compute output reference folder
    references_output_path = os.path.join(output_path, data_ref_folder)

    # compute output features folders
    features_output_folder = [ os.path.join(output_path, feature) for feature in features ]

    # compute output static folders
    statics_output_folder = [ os.path.join(output_path, static) for static in enumerate(statics_path) ]

    # concat all outputs folder
    output_folders = references_output_path + features_output_folder + static_output_folder

    output_img_index = 0

    for scene in scenes:

        # get reference image path 
        reference_image_name = os.listdir(os.path.join(references_path, scene))[0]
        reference_image_path = os.path.join(references_path, scene, reference_image_name)
        reference_image = np.array(Image.open(reference_image_path))

        # get features images list
        features_images_path = []

        for feature in features_path:
            feature_scene_path = os.path.join(feature, scene)
            feature_images = [ os.path.join(feature, img) for img in sorted(os.listdir(feature_scene_path)) ]
            features_images_path.append(feature_images)

        # get static scenes images
        static_scenes_images = []
        
        for static in statics_path:
            static_image_name = os.listdir(os.path.join(static, scene))[0]
            static_image_path = os.path.join(static, scene, static_image_name)
            static_image = np.array(Image.open(static_image_path))
            static_scenes_images.append(static_image)        

        # build path if necessary
        for output_folder in output_folders:
            scene_output_folder = os.path.join(output_folder, scene)
            if not os.path.exists(scene_output_folder):
                os.makedirs(scene_output_folder)

        # create tiles for each image
        for index in range(len(features_images_path[0])):

            images_counter = images_counter + 1

            # get shape information based on reference
            h, w, _ = reference_image.shape

            # open features image only once 
            features_images = []

            for images_path_list in features_images_path:
                feature_image = Image.open(images_path_list[index])
                features_images.append(np.array(feature_image))

            for _ in range(nb):
                
                # compute output image name (patch image name)
                output_index_str = str(output_img_index)

                while len(output_index_str) < 11:
                    output_index_str = '0' + output_index_str

                output_image_name = output_index_str + '_' + scene + '.png'


                h_random = random.randint(0, h - h_tile - 1)
                w_random = random.randint(0, w - w_tile - 1)

                h_end = h_random+h_tile
                w_end = w_random+w_tile

                # patch for reference
                tile_extract_ref = reference_image[h_random:h_end, w_random:w_end]
                output_reference_path = os.path.join(references_output_path, scene, output_image_name)
                Image.fromarray(np.array(tile_extract_ref, 'uint8')).save(output_reference_path)

                # patch for each feature
                for f_i, img in enumerate(features_images):
                    tile_extract_feature = img[h_random:h_end, w_random:w_end]
                    output_feature_path = os.path.join(features_output_folder[f_i], scene, output_image_name)
                    Image.fromarray(np.array(tile_extract_feature, 'uint8')).save(output_feature_path)

                # patch for each static data
                for s_i, img in enumerate(static_scenes_images):
                    tile_extract_static = img[h_random:h_end, w_random:w_end]
                    output_static_path = os.path.join(statics_output_folder[s_i], scene, output_image_name)
                    Image.fromarray(np.array(tile_extract_static, 'uint8')).save(output_static_path)

                output_img_index = output_img_index + 1

            # write progress using global variable
            write_progress((images_counter + 1) / number_of_images)


def main():

    global number_of_images, images_counter

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--main', type=str, help="main folder with features sub folders")
    parser.add_argument('--features', type=str, help="features to select from this main folder `" + cfg.features_list + "`", default=cfg.features_list[0])
    parser.add_argument('--static', type=str, help="list of static features to take care (managed like references)")
    parser.add_argument('--references', type=str, help='folder scenes with references')
    parser.add_argument('--nb', type=int, help='number of tile extracted from each images')
    parser.add_argument('--output', type=str, help='output folder of whole data `test` and `train` folder')
    parser.add_argument('--train_split', type=float, help='test split size of generated data (based of number of scenes)', default=0.2)

    args = parser.parse_args()

    p_main       = args.main
    p_features   = args.features.split(',')
    p_static     = args.static.split(',')
    p_references = args.references
    p_nb         = args.nb
    p_output     = args.output
    p_split      = args.train_split


    # get list scenes folders and shuffle it
    scenes_folder = os.listdir(p_references)
    random.shuffle(scenes_folder)

    nb_scenes = len(scenes_folder)
    nb_scenes_test = math.ceil(nb_scenes * p_split)

    test_scenes = scenes_folder[:nb_scenes_test]
    train_scenes = scenes_folder[nb_scenes_test:]

    print('------------------------------------------------------------------------------------------------------')
    print('Train scenes :', train_scenes)
    print('------------------------------------------------------------------------------------------------------')
    print('Test scenes :', test_scenes)

    number_of_images = sum([ len(os.listdir(scene)) for scene in os.listdir(os.path.join(p_main, p_features[0])) ]) # get total number of images from first feature path

    print('------------------------------------------------------------------------------------------------------')
    print('-- Start generating data')
    print('------------------------------------------------------------------------------------------------------')

    output_test_folder = os.path.join(p_output, data_test_folder)
    output_train_folder = os.path.join(p_output, data_train_folder)

    # contruct test tiles
    construct_tiles(test_scenes, p_main, p_features, p_static, p_references, output_test_folder, p_nb)

    # construct train tiles
    construct_tiles(train_scenes, p_main, p_features, p_static, p_references, output_train_folder, p_nb)

    # write progress using global variable
    write_progress((images_counter + 1) / number_of_images)
    

if __name__ == "__main__":
    main()