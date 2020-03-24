# dl imports
import torch

from .models import kutils
from .models.kutils import applications as apps
from .models.kutils import model_helper as mh
from .models.kutils import tensor_ops as ops
from .models.kutils import generic as gen
from .models.kutils import image_utils as img
#from tqdm import tqdm

from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

import pandas as pd
import numpy as np

import os

class Koncept512(torch.nn.Module):

    def __init__(self):
        super(Koncept512, self).__init__()

        print('--------------------------------------------------------')
        print('Initialization of Koncept loss function')
        model_name = os.getcwd() + '/losses/models/KonCept512/k512'

        pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (384, 512)), remap=False).fliplr(do=False).result)

        base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
        head = apps.fc_layers(base_model.output, name='fc', 
                                fc_sizes      = [2048, 1024, 256, 1], 
                                dropout_rates = [0.25, 0.25, 0.5, 0], 
                                batch_norm    = 2)    

        model = Model(inputs = base_model.input, outputs = head)

        gen_params = dict(batch_size  = 32, 
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

        self.model = helper.model
        self.preprocess = pre
        self.requires_grad = True
        print('--------------------------------------------------------')
        
    def forward(self, inputs, targets):
        losses = []

        # for each element in batch compute score using Koncept Model
        for i, item in enumerate(inputs):
            img_input_array = item.detach().numpy()
            img_target_array = targets[i].detach().numpy()
            
            # get shape and then reshape using channels as 3 dimension
            c, h, w = img_input_array.shape

            img_input_array = img_input_array.reshape(h, w, c)
            img_target_array = img_target_array.reshape(h, w, c)

            input_array = self.preprocess(img_input_array)
            target_array = self.preprocess(img_target_array)

            input_array = np.expand_dims(input_array, 0)
            target_array = np.expand_dims(target_array, 0)

            # get score for reference and output patch
            k_score_out = self.model.predict(input_array)[0][0]
            k_score_ref = self.model.predict(target_array)[0][0]

            # print('out', k_score_out)
            # print('ref', k_score_ref)

            # print(k_score_out / k_score_ref)
            losses.append(k_score_out / k_score_ref)

        loss_mean = torch.sum(torch.FloatTensor(losses)) / len(losses)
        loss_mean.requires_grad_()

        return loss_mean


class Koncept224(torch.nn.Module):

    def __init__(self):
        super(Koncept224, self).__init__()

        print('--------------------------------------------------------')
        print('Initialization of Koncept loss function')
        model_name = os.getcwd() + '/losses/models/KonCept224/k224'

        pre = lambda im: preprocess_fn(img.ImageAugmenter(img.resize_image(im, (224, 224)), remap=False).fliplr(do=False).result)

        base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
        head = apps.fc_layers(base_model.output, name='fc', 
                                fc_sizes      = [2048, 1024, 256, 1], 
                                dropout_rates = [0.25, 0.25, 0.5, 0], 
                                batch_norm    = 2)    

        model = Model(inputs = base_model.input, outputs = head)

        gen_params = dict(batch_size  = 32, 
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

        self.model = helper.model
        self.preprocess = pre
        self.requires_grad = True
        print('--------------------------------------------------------')
        
    def forward(self, inputs, targets):
        losses = []

        # for each element in batch compute score using Koncept Model
        for i, item in enumerate(inputs):
            img_input_array = item.detach().numpy()
            img_target_array = targets[i].detach().numpy()
            
            # get shape and then reshape using channels as 3 dimension
            c, h, w = img_input_array.shape

            img_input_array = img_input_array.reshape(h, w, c)
            img_target_array = img_target_array.reshape(h, w, c)

            input_array = self.preprocess(img_input_array)
            target_array = self.preprocess(img_target_array)

            input_array = np.expand_dims(input_array, 0)
            target_array = np.expand_dims(target_array, 0)

            # get score for reference and output patch
            k_score_out = self.model.predict(input_array)[0][0]
            k_score_ref = self.model.predict(target_array)[0][0]

            # print('out', k_score_out)
            # print('ref', k_score_ref)

            # print(k_score_out / k_score_ref)
            losses.append(k_score_out / k_score_ref)

        loss_mean = torch.sum(torch.FloatTensor(losses)) / len(losses)
        loss_mean.requires_grad_()

        return loss_mean