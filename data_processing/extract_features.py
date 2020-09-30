# main imports
import os, sys
import argparse
import random

# image processing imports
from rawls.rawls import Rawls
from rawls.stats import RawlsStats

# utils variables
import config as cfg

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


def extract(features, output_folder, scene_folder, index, images_path, extension, gamma_convert):

    # get only one time all images
    rawls_stats = RawlsStats.load(images_path)

    # compute feature using `rawls_stats`
    for feature in features:
        
        feature_path = os.path.join(output_folder, feature, scene_folder)

        rawls_stats_img = None

        if feature == 'mean':
            rawls_stats_img = rawls_stats.mean()
        
        if feature == 'variance':
            rawls_stats_img = rawls_stats.var()

        if feature == 'std':
            rawls_stats_img = rawls_stats.std()

        if feature == 'skewness':
            rawls_stats_img = rawls_stats.skew()

        if feature == 'kurtosis':
            rawls_stats_img = rawls_stats.kurtosis()

        # compute output path of feature image
        index_str = str(index)

        while len(index_str) < 5:
            index_str = "0" + index_str

        feature_image_name = scene_folder + '_' + index_str + '.' + extension
        feature_image_path = os.path.join(feature_path, feature_image_name)

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        rawls_stats_img.save(feature_image_path, gamma_convert)


def main():

    parser = argparse.ArgumentParser(description="Generate all expected features for reconstructed images for each point of view of scenes")

    parser.add_argument('--folder', type=str, help="folder scenes with pixels data (rawls files)", required=True)
    parser.add_argument('--expected', type=int, help='expected number of images into scene folder', default=10000)
    parser.add_argument('--samples', type=int, help='number of samples to use', required=True)
    parser.add_argument('--images', type=int, help='number of images for each scene', required=True)
    parser.add_argument('--ext', type=str, help='output expected extension', choices=['png', 'rawls'], required=False)
    parser.add_argument('--gamma', type=int, help='use of gamma conversion', choices=[0, 1], default=0, required=False)
    parser.add_argument('--features', type=str, help="expected features list from `" + str(cfg.features_list) + "`", default=cfg.features_list[0], required=True)
    parser.add_argument('--output', type=str, help='output folder', default='', required=True)

    args = parser.parse_args()

    p_folder   = args.folder
    p_samples  = args.samples
    p_expected = args.expected
    p_images   = args.images
    p_ext      = args.ext
    p_gamma    = bool(args.gamma)
    p_features = args.features.split(',')
    p_output   = args.output

    for feature in p_features:
        if feature not in cfg.features_list:
            raise Exception('Feature `' + feature + '` is not recognized..')

    # build output dir if not exist
    if not os.path.exists(p_output):
        os.makedirs(p_output)

    scene_folders = os.listdir(p_folder)

    # data for progress bar
    number_of_images = len(scene_folders) * p_images
    images_counter = 0

    # for each scene extract information
    for folder in scene_folders:
        folder_path = os.path.join(p_folder, folder)
        
        images_path = [ os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path)) ]

        # do not use this folder, because it is not an expected number of samples
        if len(images_path) < p_expected:
            continue

        for i in range(p_images):
            
            random.shuffle(images_path)
            images_choices = images_path[0:p_samples]

            extract(p_features, p_output, folder, i, images_choices, p_ext, p_gamma)

            # write progress using global variable
            write_progress((images_counter + 1) / number_of_images)
            images_counter += 1

    print()

if __name__ == "__main__":
    main()