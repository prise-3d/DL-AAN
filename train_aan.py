# main imports
import numpy as np
import argparse
import os

# deep learning imports
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# vizualisation
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

IMAGE_SIZE = 100
input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

BACKUP_MODEL_NAME = "synthesis_{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 20
MAX_ITERATION = 100000

# initialize weights function
def initialize_weights(arg_class):
  class_name = arg_class.__class__.__name__
  if class_name.find('Conv') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 0.0, 0.02)
  elif class_name.find('BatchNorm') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(arg_class.bias.data, 0)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(torch.nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class Encoder(torch.nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = torch.nn.Sequential(
                                       torch.nn.Conv2d(3, 32, kernel_size=4, stride=2),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(128, 256, kernel_size=4, stride=2),
                                       torch.nn.ReLU(),
                                       Flatten()
                                      )
  def forward(self, inp):
    return self.encoder(inp)


class Decoder(torch.nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = torch.nn.Sequential(
                                    UnFlatten(),
                                    torch.nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
                                    torch.nn.Sigmoid(),
                                   )
  def forward(self, inp):
    return self.decoder(inp)


class Generator(torch.nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, inp):
    return self.decoder(self.encoder(inp))


class Discriminator(torch.nn.Module):
  def __init__(self, feature_maps):
    super(Discriminator, self).__init__()
    self.feature_maps = feature_maps
    self.main = torch.nn.Sequential(torch.nn.Conv2d(3, feature_maps, 4, 2, 1, bias = False),
                                   torch.nn.LeakyReLU(0.2, inplace=True),
                                   torch.nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1, bias = False),
                                   torch.nn.BatchNorm2d(feature_maps*2),
                                   torch.nn.LeakyReLU(0.2, inplace=True),
                                   torch.nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1, bias = False),
                                   torch.nn.BatchNorm2d(feature_maps*4),
                                   torch.nn.LeakyReLU(0.2, inplace=True),
                                   torch.nn.Conv2d(feature_maps*4, feature_maps*8, 4, 2, 1, bias = False),
                                   torch.nn.BatchNorm2d(feature_maps*8),
                                   torch.nn.LeakyReLU(0.2, inplace=True),
                                   torch.nn.Conv2d(feature_maps*8, 3, 4, 1, 0, bias = False),
                                   torch.nn.Sigmoid())
  def forward(self, input_image):
    return self.main(input_image)


def main():

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--folder', type=str, help="folder scenes pixels data")
    parser.add_argument('--output', type=str, help='output model name')
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=30)

    args = parser.parse_args()

    p_folder     = args.folder
    p_output     = args.output
    p_batch_size = args.batch_size
    p_epochs     = args.epochs

    learning_rate = 0.0002

    # build data path
  
    # TODO : prepare train  (do h_flip, v_flip)
    # train_noises_h_flip = torchvision.datasets.ImageFolder('/content/drive/My Drive/faces94/female', transform=transforms.Compose([
    #     transforms.RandomHorizontalFlip(1.), # flip horizontally all images
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ]))

    # train_noises_v_flip = torchvision.datasets.ImageFolder('/content/drive/My Drive/faces94/female', transform=transforms.Compose([
    #     transforms.RandomVerticalFlip(1.), # flip horizontally all images
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ]))

    train_path = os.path.join(p_folder, 'train')
    noises_train_path = os.path.join(train_path, 'noises')
    references_train_path = os.path.join(train_path, 'references')

    train_noises = torchvision.datasets.ImageFolder(noises_train_path, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(1.), # flip horizontally all images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_ref = torchvision.datasets.ImageFolder(references_train_path, transform=transforms.Compose([
        transforms.RandomVerticalFlip(1.), # flip horizontally all images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))


    # Dataloader = [torch.utils.data.DataLoader(female, batch_size=128, shuffle=True, num_workers=8), 
    #           torch.utils.data.DataLoader(male, batch_size=128, shuffle=True, num_workers=8), 
    #           torch.utils.data.DataLoader(malestaff, batch_size=128, shuffle=True, num_workers=8)]

    DataLoaderNoises = torch.utils.data.DataLoader(train_noises, batch_size=32, shuffle=False, num_workers=0)
    DataLoaderRef = torch.utils.data.DataLoader(train_ref, batch_size=32, shuffle=False, num_workers=0)

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # define models and loss functions
    encoder = Encoder().to(device)
    print(encoder)

    decoder = Decoder().to(device)
    print(decoder)

    parameters = list(encoder.parameters())+ list(decoder.parameters())
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    discriminator = Discriminator(32).to(device)
    discriminator.apply(initialize_weights)
    print(discriminator)

    # define writer
    writer = SummaryWriter()

    iteration = 0

    for epoch in range(150):
            
        batchListRef = list(enumerate(DataLoaderRef, 0))

        for batch_id, noisy_data in enumerate(DataLoaderNoises, 0):
            
            batch_noises, _ = noisy_data
            batch_ref, _ = batchListRef[batch_id]

            optimizer.zero_grad()
          
            print(np.array(batch_noises).shape)
            # constuct new images 
            output = encoder(batch_noises)
            output = decoder(output)

            # pass expected ref 
            gained_loss = loss_func(output, batch_ref)
            
            # mean loss
            autoencoder_total_loss += gained_loss.mean().item()

            print("Loss " + autoencoder_total_loss)

            gained_loss.backward()
            optimizer.step()

            if iteration % REPORT_EVERY_ITER == 0:
                #log.info("Iter %d: _loss=%.3e, dis_loss=%.3e", iteration, autoencoder_total_loss, 0)
                writer.add_scalar("gen_loss", autoencoder_total_loss, iteration)
                #writer.add_scalar("dis_loss", np.mean(dis_losses), iteration)
                
            if iteration % SAVE_IMAGE_EVERY_ITER == 0:
                  writer.add_image("fake", vutils.make_grid(output.data[:IMAGE_SIZE], normalize=True), iteration)
                  writer.add_image("real", vutils.make_grid(batch_ref.data[:IMAGE_SIZE], normalize=True), iteration)

            iteration += 1
        print("EPOCH:", epoch+1)
        #print("AVERAGE LOSS:", total_loss)

if __name__ == "__main__":
    main()