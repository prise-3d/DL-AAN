import kutils
from kutils import applications as apps
from kutils import model_helper as mh
from kutils import tensor_ops as ops
from kutils import generic as gen
from kutils import image_utils as img
#from tqdm import tqdm

from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

import pandas as pd
import numpy as np

import sys, os
import argparse

def run224(images_path):

    model_name = os.getcwd() + '/losses/models/KonCept224/k224'

    pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (224, 224)), remap=False).fliplr(do=False).result)

    base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
    head = apps.fc_layers(base_model.output, name='fc', 
                            fc_sizes      = [2048, 1024, 256, 1], 
                            dropout_rates = [0.25, 0.25, 0.5, 0], 
                            batch_norm    = 2)    

    model = Model(inputs = base_model.input, outputs = head)

    gen_params = dict(batch_size  = 32, 
                        data_path   = images_path,
                        process_fn  = pre, 
                        input_shape = (224,224,3),
                        outputs     = ('MOS',))

    # Wrapper for the model, helps with training and testing
    helper = mh.ModelHelper(model, 'KonCept224', pd.DataFrame(), 
                            loss='MSE', metrics=["MAE", ops.plcc_tf],
                            monitor_metric = 'val_loss', 
                            monitor_mode   = 'min', 
                            multiproc   = True, workers = 5,
                            gen_params  = gen_params)

    helper.load_model(model_name=model_name)

    for one_image in sorted(os.listdir(images_path)):

        if '.png' in one_image:
            # load, pre-process it, and pass it to the model
            image_full_path = os.path.join(images_path, one_image)
            one_img = pre(img.read_image(image_full_path))
            one_img = np.expand_dims(one_img, 0)
            one_img_score = helper.model.predict(one_img)
            print(one_image, '=>', one_img_score[0][0])

def run512(images_path):

    model_name = os.getcwd() + '/losses/models/KonCept512/k512'

    pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (384, 512)), remap=False).fliplr(do=False).result)

    base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
    head = apps.fc_layers(base_model.output, name='fc', 
                            fc_sizes      = [2048, 1024, 256, 1], 
                            dropout_rates = [0.25, 0.25, 0.5, 0], 
                            batch_norm    = 2)    

    model = Model(inputs = base_model.input, outputs = head)

    gen_params = dict(batch_size  = 32, 
                        data_path   = images_path,
                        process_fn  = pre, 
                        input_shape = (384,512,3),
                        outputs     = ('MOS',))

    # Wrapper for the model, helps with training and testing
    helper = mh.ModelHelper(model, 'KonCept512', pd.DataFrame(), 
                            loss='MSE', metrics=["MAE", ops.plcc_tf],
                            monitor_metric = 'val_loss', 
                            monitor_mode   = 'min', 
                            multiproc   = True, workers = 5,
                            gen_params  = gen_params)

    helper.load_model(model_name=model_name)

    for one_image in sorted(os.listdir(images_path)):

        if '.png' in one_image:
            # load, pre-process it, and pass it to the model
            image_full_path = os.path.join(images_path, one_image)
            one_img = pre(img.read_image(image_full_path))
            one_img = np.expand_dims(one_img, 0)
            one_img_score = helper.model.predict(one_img)
            print(one_image, '=>', one_img_score[0][0])
            # new_image_name = os.path.splitext(image_full_path)[0] + '_scoreK512_' + str(one_img_score[0][0]) + os.path.splitext(image_full_path)[1]
            # os.rename(image_full_path, new_image_name)


def main():

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--folder', type=str, help="folder with images to predict")
    parser.add_argument('--model', type=int, help="KonCept model choice", choices=[224, 512])

    args = parser.parse_args()

    p_folder = args.folder
    p_model = args.model

    if p_model == 224:
        run224(p_folder)
    if p_model == 512:
        run512(p_folder)

if __name__ == "__main__":
    main()