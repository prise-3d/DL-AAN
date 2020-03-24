import torch

from pytorch_ssim import ssim

class SSIM(torch.nn.Module):

    def __init__(self):
        super(SSIM, self).__init__()
        self.loss = ssim

    def forward(self, output, reference):
        return 1. - self.loss(output, reference)