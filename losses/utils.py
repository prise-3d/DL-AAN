import torch 

# from .lmse import LMSE
# from .ssim import SSIM
# from .koncept import Koncept512, Koncept224
# from .mixin import MSEK512, MSEK224

# loss list
loss_choices = ['mse', 'ssim', 'bce', 'koncept512', 'koncept224', 'msek512', 'msek224', 'lmse', 'L1']

def instanciate(choice):
    
    if choice not in loss_choices:
        raise Exception('invalid loss function choice')

    if choice == 'L1':
        return torch.nn.L1Loss()

    if choice == 'mse':
        return torch.nn.MSELoss()

    # if choice == 'ssim':
    #     return SSIM()

    if choice == 'bce':
        return torch.nn.BCELoss()

    # if choice == 'koncept512':
    #     return Koncept512()

    # if choice == 'koncept224':
    #     return Koncept224()

    # if choice == 'msek512':
    #     return MSEK512()

    # if choice == 'msek224':
    #     return MSEK224()

    # if choice == 'lmse':
    #     return LMSE()