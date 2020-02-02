# main imports
import numpy as np
import argparse

# deep learning imports
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt



def initialize_weights(arg_class):
  class_name = arg_class.__class__.__name__
  if class_name.find('Conv') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 0.0, 0.02)
  elif class_name.find('BatchNorm') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(arg_class.bias.data, 0)

class Encoder(torch.nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = torch.nn.Sequential(
                                       torch.nn.Conv2d(3, 32, 4, 2, 1, bias = False),
                                       torch.nn.LeakyReLU(0.2, inplace=True), 
                                       torch.nn.Conv2d(32, 64, 4, 2, 1, bias = False),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                                       torch.nn.BatchNorm2d(128),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                                       torch.nn.BatchNorm2d(256),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                                       torch.nn.BatchNorm2d(512),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.Conv2d(512, 100, 4, 1, 0, bias = False),
                                      )
  def forward(self, inp):
    return self.encoder(inp)

class Decoder(torch.nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = torch.nn.Sequential(
                                   torch.nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                   torch.nn.BatchNorm2d(512),
                                   torch.nn.ReLU(True),
                                   torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                   torch.nn.BatchNorm2d(256),
                                   torch.nn.ReLU(True),
                                   torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                   torch.nn.BatchNorm2d(128),
                                   torch.nn.ReLU(True),
                                   torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                   torch.nn.BatchNorm2d(64),
                                   torch.nn.ReLU(True),
                                   torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                   torch.nn.BatchNorm2d(32),
                                   torch.nn.ReLU(True),
                                   torch.nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
                                   torch.nn.Tanh()
                                   )
  def forward(self, inp):
    return (self.decoder(inp))

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


    # TODO : prepare train data
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    generator = Generator().to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    print(generator)


    discriminator = Discriminator(32).to(device)
    discriminator.apply(initialize_weights)
    print(discriminator)


    iterations = 0

    generator.to(device)
    for epoch in range(150):
        for j in range(len(Dataloader)):
            total_loss = 0
            for i, data in enumerate(Dataloader[j], 0):
                optimizer.zero_grad()
                batch = data[0].to(device)
                batch_size = batch.size(0)
                gained_image = generator(batch)
                output = encoder(image_n)
                output = decoder(output)
                loss = loss_func(output,image)
                gained_loss=loss(gained_image, batch)
                
                
                total_loss += gained_loss.mean().item()

                gained_loss.backward()
                optimizer.step()
        print("EPOCH:", epoch+1)
        print("AVERAGE LOSS:", total_loss)

if __name__ == "__main__":
    main()