# main imports
import argparse
import numpy as np
import os, sys

# image imports
from PIL import Image
from rawls.rawls import Rawls

# others imports
import math
import random

# modules imports
import config as cfg

# important variables
data_train_folder  = 'train'
data_test_folder   = 'test'

data_ref_folder    = cfg.references_folder

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
Preprocessing for samples based values
'''
def array_preprocessing(array, display=False):

    array = np.log10(array + 1)
    log_min = np.min(array)
    log_max = np.max(array)

    if display:
        print(np.min(log_min))
        print(np.max(log_max))

    for dim in range(array.ndim):
        v = array[:, :, dim]
        array[:, :, dim] = (v - log_min) / (log_max - log_min)

    return array

'''
Constuct all dataset with specific tile size
'''
def construct_tiles(scenes, main_path, features, statics_path, references_path, output_path, nb, tile_size):

    global images_counter

    h_tile, w_tile = tile_size

    # compute features path
    features_path = [ os.path.join(main_path, feature) for feature in features ]

    # compute output reference folder
    references_output_path = os.path.join(output_path, data_ref_folder)

    # compute output features folders
    features_output_folder = [ os.path.join(output_path, feature) for feature in features ]

    # compute output static folders
    statics_output_folder = []

    for static in statics_path:
        path_split = static.split('/')

        if path_split[-1] != '':
            statics_output_folder.append(os.path.join(output_path, path_split[-1]))
        else:
            statics_output_folder.append(os.path.join(output_path, path_split[-2]))

    # concat all outputs folder
    output_folders = [references_output_path] + features_output_folder + statics_output_folder
    print(output_folders)

    output_img_index = 0

    for scene in scenes:

        # get reference image path 
        reference_image_name = os.listdir(os.path.join(references_path, scene))[0]
        reference_image_path = os.path.join(references_path, scene, reference_image_name)
        reference_image = Rawls.load(reference_image_path)

        # add preprocessing step for samples based input
        # processing
        reference_image = array_preprocessing(reference_image.data, True)

        # get features images list
        features_images_path = []

        for feature in features_path:
            feature_scene_path = os.path.join(feature, scene)
            feature_images = [ os.path.join(feature, scene, img) for img in sorted(os.listdir(feature_scene_path)) ]
            features_images_path.append(feature_images)

        # get static scenes images
        static_scenes_images = []
        
        for static in statics_path:
            static_image_name = os.listdir(os.path.join(static, scene))[0]
            static_image_path = os.path.join(static, scene, static_image_name)

            static_image = Rawls.load(static_image_path)
            static_image = static_image.normalize()
            static_scenes_images.append(static_image.data)        

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
                feature_image = Rawls.load(images_path_list[index])
                # add preprocessing step for samples based input

                # processing
                feature_image.data = array_preprocessing(feature_image.data)

                features_images.append(feature_image.data)

            for _ in range(nb):
                
                # compute output image name (patch image name)
                output_index_str = str(output_img_index)

                while len(output_index_str) < 11:
                    output_index_str = '0' + output_index_str

                output_image_name = scene + '_' + output_index_str + '.npy'


                h_random = random.randint(0, h - h_tile - 1)
                w_random = random.randint(0, w - w_tile - 1)

                h_end = h_random+h_tile
                w_end = w_random+w_tile

                # patch for reference
                tile_extract_ref = reference_image[h_random:h_end, w_random:w_end]
                output_reference_path = os.path.join(references_output_path, scene, output_image_name)

                # tile_extract_ref = array_preprocessing(tile_extract_ref)

                np.save(output_reference_path, np.array(tile_extract_ref, 'float32'))

                # patch for each feature
                for f_i, img in enumerate(features_images):
                    tile_extract_feature = img[h_random:h_end, w_random:w_end]
                    output_feature_path = os.path.join(features_output_folder[f_i], scene, output_image_name)

                    # tile_extract_feature = array_preprocessing(tile_extract_feature)
                    
                    np.save(output_feature_path, np.array(tile_extract_feature, 'float32'))

                # patch for each static data
                for s_i, img in enumerate(static_scenes_images):
                    tile_extract_static = img[h_random:h_end, w_random:w_end]
                    output_static_path = os.path.join(statics_output_folder[s_i], scene, output_image_name)
                    np.save(output_static_path, np.array(tile_extract_static, 'float32'))

                output_img_index = output_img_index + 1

            # write progress using global variable
            write_progress((images_counter + 1) / number_of_images)

def main():

    global number_of_images, images_counter

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--main', type=str, help="main folder with features sub folders")
    parser.add_argument('--features', type=str, help="features to select from this main folder `" + str(cfg.features_list) + "`", default=cfg.features_list[0])
    parser.add_argument('--statics', type=str, help="list of static features to take care (managed like references)", default='')
    parser.add_argument('--references', type=str, help='folder scenes with references')
    parser.add_argument('--nb', type=int, help='number of tile extracted from each images')
    parser.add_argument('--tile_size', type=str, help='specify size of the tile used', default='32,32')
    parser.add_argument('--output', type=str, help='output folder of whole data `test` and `train` folder')
    parser.add_argument('--train_split', type=float, help='test split size of generated data (based of number of scenes)', default=0.2)

    args = parser.parse_args()

    p_main       = args.main
    p_features   = args.features.split(',') if len(args.features) > 0 else []
    p_statics    = args.statics.split(',') if len(args.statics) > 0 else []
    p_references = args.references
    p_nb         = args.nb
    p_tile       = args.tile_size.split(',')
    p_output     = args.output
    p_split      = args.train_split

    tile_size = int(p_tile[0]), int(p_tile[1])

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

    number_of_images = sum([ len(os.listdir(os.path.join(p_main, p_features[0], scene))) for scene in os.listdir(os.path.join(p_main, p_features[0])) ]) # get total number of images from first feature path

    print('------------------------------------------------------------------------------------------------------')
    print('-- Start generating data')
    print('------------------------------------------------------------------------------------------------------')

    output_test_folder = os.path.join(p_output, data_test_folder)
    output_train_folder = os.path.join(p_output, data_train_folder)

    # contruct test tiles
    construct_tiles(test_scenes, p_main, p_features, p_statics, p_references, output_test_folder, p_nb, tile_size)

    # construct train tiles
    construct_tiles(train_scenes, p_main, p_features, p_statics, p_references, output_train_folder, p_nb, tile_size)

    print()
    

if __name__ == "__main__":
    main()