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

# Concatenate noisy and reference data
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Encoder(torch.nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = torch.nn.Sequential(
                                       torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       #torch.nn.MaxPool2d(3, stride=2, padding=1),
                                       torch.nn.BatchNorm2d(32),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       #torch.nn.MaxPool2d(3, stride=2, padding=1),
                                       torch.nn.BatchNorm2d(64),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(128),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(256),
                                       torch.nn.Dropout(0.3),
                                      )
  def forward(self, inp):
    # print('Encoder input', inp.size())
    # x = inp
    # for id, layer in enumerate(self.encoder):
    #     x = layer(x)
    #     print('Layer', id, x.size())
    return self.encoder(inp)


class Decoder(torch.nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = torch.nn.Sequential(
                                    torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm2d(128),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm2d(64),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm2d(32),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                                    torch.nn.Sigmoid(),
                                   )
  def forward(self, inp):
    # print('Decoder input', inp.size())
    # x = inp
    # for id, layer in enumerate(self.decoder):
    #     x = layer(x)
    #     print('Layer', id, x.size())
    return self.decoder(inp)


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
                                    torch.nn.Conv2d(in_channels=feature_maps * 8, out_channels=feature_maps * 16, kernel_size=4, stride=1, padding=1),
                                    torch.nn.BatchNorm2d(DISCR_FILTERS * 16),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(in_channels=feature_maps * 16, out_channels=1, kernel_size=3, stride=2, padding=0),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim


def main():

    save_model = False
    load_model = False
    restart = False
    start_epoch = 0
    start_iteration = 0

    parser = argparse.ArgumentParser(description="Output data file")

    parser.add_argument('--folder', type=str, help="folder scenes pixels data")
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=100)
    parser.add_argument('--start_discriminator', type=int, help='number of epochs when you want the discriminator started', default=5)
    parser.add_argument('--save', type=str, help='save folder for backup model', default='')
    parser.add_argument('--load', type=str, help='folder backup model', default='')

    args = parser.parse_args()

    p_folder              = args.folder
    p_batch_size          = args.batch_size
    p_epochs              = args.epochs
    p_start_discriminator = args.start_discriminator
    p_save                = args.save
    p_load                = args.load

    if len(p_load) > 0:
        load_model = True

    if len(p_save) > 0:
        save_model = True

    # build data path
    train_path = os.path.join(p_folder, 'train')
    noises_train_path = os.path.join(train_path, 'noises')
    references_train_path = os.path.join(train_path, 'references')

    # shuffle data loader and made possible to keep track well of reference
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(
            torchvision.datasets.ImageFolder(noises_train_path, transform=transforms.Compose([
            #transforms.RandomVerticalFlip(1.), # flip horizontally all images
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])),
            torchvision.datasets.ImageFolder(references_train_path, transform=transforms.Compose([
            #transforms.RandomVerticalFlip(1.), # flip horizontally all images
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]))
        ),
        batch_size=p_batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    train_dataset_batch_size = len(train_loader)
             
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

    print('autoencoder total parameters : ', sum(p.numel() for p in autoencoder_parameters))

    ####################################
    # Discriminator model declaration  #
    ####################################
    discriminator = Discriminator(32).to(device)
    #discriminator.apply(initialize_weights)

    discriminator_loss_func = torch.nn.BCELoss()
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    print(discriminator)
    print('discriminator total parameters : ', sum(p.numel() for p in discriminator.parameters()))

    print('--------------------------------------------------------')
    print("Train data loader size : ", train_dataset_batch_size)

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

        if not os.path.exists(global_model_path):
            print('-------------------------')
            print('Model backup not found...')
            print('-------------------------')
        else:
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
            start_iteration = backup_iteration
            start_epoch = backup_epochs
            restart = True

            print('---------------------------')
            print('Model backup found....')
            print('Restart from epoch', start_epoch)
            print('Restart from iteration', start_iteration)
            print('---------------------------')
        
    # define writer
    writer = SummaryWriter()

    for epoch in range(p_epochs):
            
        # initialize correct detected from discriminator
        correct_detected = 0
        print(iteration)

        # check dataset in order to restart
        if train_dataset_batch_size * (epoch + 1) > start_iteration and restart:
            iteration += train_dataset_batch_size
            print(iteration)
            continue

        # if needed to restart, then restart from expected train_loader element
        if restart:
            nb_viewed_elements = start_iteration % train_dataset_batch_size
            train_dataset = list(train_loader)[nb_viewed_elements:]
            restart = False
        else:
            train_dataset = train_loader


        for batch_id, (input_data, target_data)  in enumerate(train_dataset):
            
            if start_iteration > iteration:
                iteration += 1
                continue

            # 1. get noises batch and reference
            batch_noises, _ = input_data
            batch_ref_data, _ = target_data
            
            # 2. Train autoencoder..
            autoencoder_optimizer.zero_grad()

            output = encoder(batch_noises)
            output = decoder(output)

            autoencoder_loss = autoencoder_loss_func(output, batch_ref_data)
            autoencoder_losses.append(autoencoder_loss.item())

            autoencoder_loss.backward()
            autoencoder_optimizer.step()

            # 3. train discriminator
            # only if necessary (generator trained well before) - assumption: avoid of local optima
            if epoch >= p_start_discriminator:
                discriminator_optimizer.zero_grad()

                discriminator_output_true_v = discriminator(batch_ref_data)
                discriminator_output_fake_v = discriminator(output.detach())

                nb_true_output = len(discriminator_output_true_v)
                nb_fake_output = len(discriminator_output_fake_v)

                current_true_label = true_labels_v[:nb_true_output]
                current_fake_label = fake_labels_v[:nb_fake_output]

                discriminator_loss = discriminator_loss_func(discriminator_output_true_v, current_true_label) + discriminator_loss_func(discriminator_output_fake_v, current_fake_label)
                discriminator_losses.append(discriminator_loss.item())

                discriminator_loss.backward()
                discriminator_optimizer.step()
                
                # 4. compute accuracy from the epoch
                discriminator_output_true = (discriminator_output_true_v > 0.5).float()
                discriminator_output_fake = (discriminator_output_fake_v > 0.5).float()

                correct_detected += (discriminator_output_true == current_true_label).float().sum() + (discriminator_output_fake == current_fake_label).float().sum()
                discriminator_accuracy = correct_detected / float(((batch_id + 1) * p_batch_size * 2))

            # 5. Add to summary writer tensorboard
            if iteration % REPORT_EVERY_ITER == 0:

                # save only if necessary (generator trained well)
                if epoch >= p_start_discriminator:
                    log.info("Iteration %d: autoencoder_loss=%.3e, discriminator_loss=%.3e, discriminator_accuracy=%.3f", iteration, np.mean(autoencoder_losses), np.mean(discriminator_losses), discriminator_accuracy)
                else:
                    log.info("Iteration %d: autoencoder_loss=%.3e", iteration, np.mean(autoencoder_losses))
                
                writer.add_scalar("autoencoder_loss", np.mean(autoencoder_losses), iteration)

                # save only if necessary (generator trained well)
                if epoch >= p_start_discriminator:
                    writer.add_scalar("discriminator_loss", np.mean(discriminator_losses), iteration)
                    writer.add_scalar("discriminator_acc", discriminator_accuracy, iteration)

                autoencoder_losses = []
                discriminator_losses = []
                
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

                # save only if necessary (generator trained well)
                if epoch >= p_start_discriminator:
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
                    
        if epoch >= start_epoch:
            writer.add_scalar("epoch", epoch + 1, iteration)

if __name__ == "__main__":
    main()