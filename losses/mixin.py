# dl imports
import torch

from .koncept import Koncept224, Koncept512

#from tqdm import tqdm

from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

import pandas as pd
import numpy as np

import os

class MSEK512(torch.nn.Module):

    def __init__(self):
        super(MSEK512, self).__init__()

        self.loss1 = torch.nn.MSELoss()
        self.loss2 = Koncept512()
        
    def forward(self, inputs, targets):
        
        total_loss = self.loss1(inputs, targets) * self.loss2(inputs, targets)
        print(total_loss)
        return total_loss


class MSEK224(torch.nn.Module):

    def __init__(self):
        super(MSEK224, self).__init__()

        self.loss1 = torch.nn.MSELoss()
        self.loss2 = Koncept224()
        
    def forward(self, inputs, targets):

        total_loss = self.loss1(inputs, targets) * self.loss2(inputs, targets)
        print(total_loss)
        return total_loss