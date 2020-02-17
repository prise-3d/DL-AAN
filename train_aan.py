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

# logger import
import gym
log = gym.logger
log.set_level(gym.logger.INFO)

# other parameters
IMAGE_SIZE = 64
DISCR_FILTERS = 32

BACKUP_MODEL_NAME = "{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0002
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
                                       torch.nn.Conv2d(3, 32, kernel_size=3, stride=1),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                       torch.nn.ReLU(),
                                       #Flatten()
                                      )
  def forward(self, inp):
    return self.encoder(inp)


class Decoder(torch.nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = torch.nn.Sequential(
                                    #UnFlatten(),
                                    torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
                                    torch.nn.ReLU(),
                                    torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1),
                                    torch.nn.Sigmoid(),
                                   )
  def forward(self, inp):
    return self.decoder(inp)


# class Generator(torch.nn.Module):
#   def __init__(self):
#     super(Generator, self).__init__()
#     self.encoder = Encoder()
#     self.decoder = Decoder()

#   def forward(self, inp):
#     return self.decoder(self.encoder(inp))


class Discriminator(torch.nn.Module):
  def __init__(self, feature_maps):
    super(Discriminator, self).__init__()
    self.feature_maps = feature_maps
    self.main = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=feature_maps, kernel_size=3, stride=1, padding=1),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(in_channels=feature_maps, out_channels=feature_maps*2, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(DISCR_FILTERS*2),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(in_channels=feature_maps * 2, out_channels=feature_maps * 4, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(DISCR_FILTERS * 4),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(in_channels=feature_maps * 4, out_channels=feature_maps * 8, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(DISCR_FILTERS * 8),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(in_channels=feature_maps * 8, out_channels=1, kernel_size=3, stride=2, padding=0),
                                    torch.nn.Sigmoid())
  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim


def main():

    save_model = False
    load_model = False
    start_epoch = 0

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--folder', type=str, help="folder scenes pixels data")
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=100)
    parser.add_argument('--save', type=str, help='save folder for backup model', default='')
    parser.add_argument('--load', type=str, help='folder backup model', default='')

    args = parser.parse_args()

    p_folder     = args.folder
    p_batch_size = args.batch_size
    p_epochs     = args.epochs
    p_save       = args.save
    p_load       = args.load

    if len(p_load) > 0:
        load_model = True

    if len(p_save) > 0:
        save_model = True

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
        #transforms.RandomHorizontalFlip(1.), # flip horizontally all images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    train_ref = torchvision.datasets.ImageFolder(references_train_path, transform=transforms.Compose([
        #transforms.RandomVerticalFlip(1.), # flip horizontally all images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))


    # Dataloader = [torch.utils.data.DataLoader(female, batch_size=128, shuffle=True, num_workers=8), 
    #           torch.utils.data.DataLoader(male, batch_size=128, shuffle=True, num_workers=8), 
    #           torch.utils.data.DataLoader(malestaff, batch_size=128, shuffle=True, num_workers=8)]

    DataLoaderNoises = torch.utils.data.DataLoader(train_noises, batch_size=p_batch_size, shuffle=False, num_workers=0)
    DataLoaderRef = torch.utils.data.DataLoader(train_ref, batch_size=p_batch_size, shuffle=False, num_workers=0)

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ##################################
    # Autoencoder model declaration  #
    ##################################
    # define models and loss functions
    encoder = Encoder().to(device)
    print(encoder)

    decoder = Decoder().to(device)
    print(decoder)

    # set autoencoder parameters
    autoencoder_parameters = list(encoder.parameters()) + list(decoder.parameters())

    autoencoder_loss_func = torch.nn.MSELoss()
    autoencoder_optimizer = torch.optim.Adam(autoencoder_parameters, lr=LEARNING_RATE, betas=(0.5, 0.999))

    ####################################
    # Discriminator model declaration  #
    ####################################
    discriminator = Discriminator(32).to(device)
    discriminator.apply(initialize_weights)

    discriminator_loss_func = torch.nn.BCELoss()
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    print(discriminator)

    # default params
    iteration = 0
    autoencoder_losses = []
    discriminator_losses = []

    # declare usefull variables for discriminator loss function
    true_labels_v = torch.ones(p_batch_size, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(p_batch_size, dtype=torch.float32, device=device)

    # prepare folder names to save models
    if save_model:
        models_folder_path = os.path.join(BACKUP_FOLDER, p_save)

        global_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('global'))
        discriminator_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
        autoencoder_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    if load_model:
        models_folder_path = os.path.join(BACKUP_FOLDER, p_load)

        global_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('global'))
        discriminator_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
        autoencoder_model_path = os.path.join(models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    # load models checkpoint if exists
    if load_model:

        # load autoencoder state
        autoencoder_checkpoint = torch.load(autoencoder_model_path)

        encoder.load_state_dict(autoencoder_checkpoint['encoder_state_dict'])
        decoder.load_state_dict(autoencoder_checkpoint['decoder_state_dict'])
        autoencoder_optimizer.load_state_dict(autoencoder_checkpoint['optimizer_state_dict'])
        autoencoder_losses = autoencoder_checkpoint['autoencoder_losses']

        # load discriminator state
        discriminator_checkpoint = torch.load(discriminator_model_path)

        discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
        discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
        discriminator_losses = discriminator_checkpoint['discriminator_losses']

        # load global state
        global_checkpoint = torch.load(global_model_path)

        backup_iteration = global_checkpoint['iteration']
        backup_epochs = global_checkpoint['epochs'] 

        # update context variables
        iteration = backup_iteration
        start_epoch = backup_epochs
        
    # define writer
    writer = SummaryWriter()

    for epoch in range(p_epochs):
            
        if epoch < start_epoch:
            continue 
        
        # initialize correct detected from discriminator
        correct_detected = 0

        # prepare enumerate for batch list ref also
        batchListRef = list(enumerate(DataLoaderRef, 0))

        for batch_id, noisy_data in enumerate(DataLoaderNoises, 0):
            
            # 1. get noises batch and reference
            batch_noises, _ = noisy_data
            _, batch_ref = batchListRef[batch_id]

            batch_ref_data, _ = batch_ref
            
            # 2. Train autoencoder..
            autoencoder_optimizer.zero_grad()

            output = encoder(batch_noises)
            output = decoder(output)

            autoencoder_loss = autoencoder_loss_func(output, batch_ref_data)
            autoencoder_losses.append(autoencoder_loss.item())

            autoencoder_loss.backward()
            autoencoder_optimizer.step()

            # 3. train discriminator
            discriminator_optimizer.zero_grad()

            discriminator_output_true_v = discriminator(batch_ref_data)
            discriminator_output_fake_v = discriminator(output.detach())

            discriminator_loss = discriminator_loss_func(discriminator_output_true_v, true_labels_v) + discriminator_loss_func(discriminator_output_fake_v, fake_labels_v)
            discriminator_losses.append(discriminator_loss.item())

            discriminator_loss.backward()
            discriminator_optimizer.step()
            
            # 4. compute accuracy from the epoch
            discriminator_output_true = (discriminator_output_true_v > 0.5).float()
            discriminator_output_fake = (discriminator_output_fake_v > 0.5).float()

            correct_detected += (discriminator_output_true == true_labels_v).float().sum() + (discriminator_output_fake == fake_labels_v).float().sum()
            #print(correct_detected)
            discriminator_accuracy = correct_detected / float(((batch_id + 1) * p_batch_size * 2))

            # 5. Add to summary writer tensorboard
            if iteration % REPORT_EVERY_ITER == 0:
                log.info("Iteration %d: autoencoder_loss=%.3e, discriminator_loss=%.3e, discriminator_accuracy=%.3f", iteration, np.mean(autoencoder_losses), np.mean(discriminator_losses), discriminator_accuracy)
                
                writer.add_scalar("autoencoder_loss", np.mean(autoencoder_losses), iteration)
                writer.add_scalar("discriminator_loss", np.mean(discriminator_losses), iteration)
                writer.add_scalar("discriminator_acc", discriminator_accuracy, iteration)

                autoencoder_losses = []
                
            if iteration % SAVE_IMAGE_EVERY_ITER == 0:
                writer.add_image("fake", vutils.make_grid(output.data[:IMAGE_SIZE], normalize=True), iteration)
                writer.add_image("noisy", vutils.make_grid(batch_noises[:IMAGE_SIZE], normalize=True), iteration)
                writer.add_image("real", vutils.make_grid(batch_ref_data[:IMAGE_SIZE], normalize=True), iteration)

            # 6. Backup models information
            if iteration % BACKUP_EVERY_ITER == 0:
                if not os.path.exists(models_folder_path):
                    os.makedirs(models_folder_path)

                torch.save({
                            'iteration': iteration,
                            'encoder_state_dict': encoder.state_dict(),
                            'decoder_state_dict': decoder.state_dict(),
                            'optimizer_state_dict': autoencoder_optimizer.state_dict(),
                            'autoencoder_losses': autoencoder_losses
                        }, autoencoder_model_path)

                torch.save({
                            'model_state_dict': discriminator.state_dict(),
                            'optimizer_state_dict': discriminator_optimizer.state_dict(),
                            'discriminator_losses': discriminator_losses
                    }, discriminator_model_path)

                torch.save({
                            'iteration': iteration,
                            'epochs': epoch
                        }, global_model_path)

            # 7. increment number of iteration
            iteration += 1
                    
        print("EPOCH:", epoch + 1)

if __name__ == "__main__":
    main()