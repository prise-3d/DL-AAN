import torch

from skimage.metrics import mean_squared_error
from ipfml.processing.transform import get_LAB_L

class LMSE(torch.nn.Module):

    def __init__(self):
        super(LMSE, self).__init__()

    def forward(self, inputs, targets):

        losses = []

        # for each element in batch compute score using Koncept Model
        for i, item in enumerate(inputs):
            img_input_array = item.detach().numpy()
            img_target_array = targets[i].detach().numpy()
            
            # get shape and then reshape using channels as 3 dimension
            c, h, w = img_input_array.shape

            # img_input_array = get_LAB_L(img_input_array.reshape(h, w, c))
            # img_target_array = get_LAB_L(img_target_array.reshape(h, w, c))

            img_input_array = (img_input_array.reshape(h, w, c))
            img_target_array = (img_target_array.reshape(h, w, c))

            # multichanel is applied on last dimension
            score = mean_squared_error(img_input_array, img_target_array)

            losses.append(score)

        loss_mean = torch.sum(torch.FloatTensor(losses)) / len(losses)
        loss_mean.requires_grad_()

        return loss_mean