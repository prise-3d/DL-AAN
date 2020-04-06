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
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# models imports
from models.downsampling import Encoder, Decoder, Discriminator

# losses imports
from losses.utils import instanciate, loss_choices

# logger import
import gym
log = gym.logger
log.set_level(gym.logger.INFO)

# other parameters
IMAGE_SIZE = 64

BACKUP_MODEL_NAME = "{}_model.pt"
BACKUP_FOLDER = "saved_models"
BACKUP_EVERY_ITER = 1

LEARNING_RATE = 0.0002
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 20

def get_reference_and_inputs(data):

    # reference batch data
    batch_ref_data, _ = data[0]

    # compute others inputs data
    others_inputs = data[1:]

    batch_list = []

    for input_data in others_inputs:
        batch_input, _ = input_data
        batch_list.append(batch_input)

    batch_inputs = torch.cat(batch_list, dim=1)

    return batch_ref_data, batch_inputs

# initialize weights function
def initialize_weights(arg_class):
  class_name = arg_class.__class__.__name__
  if class_name.find('Conv') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 0.0, 0.02)
  elif class_name.find('BatchNorm') != -1:
    torch.nn.init.normal_(arg_class.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(arg_class.bias.data, 0)


# Concatenate features and reference data
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def main():

    save_model = False
    load_model = False
    restart = False
    start_epoch = 0
    start_iteration = 0

    parser = argparse.ArgumentParser(description="Generate model using specific features")

    parser.add_argument('--folder', type=str, help="folder with train/test folders within all features sub folders")
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=32)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=100)
    parser.add_argument('--autoencoder_loss', type=str, help="loss function choice", choices=loss_choices, default='mse')
    parser.add_argument('--discriminator_loss', type=str, help="loss function choice", choices=loss_choices, default='bce')
    parser.add_argument('--start_discriminator', type=int, help='number of epochs when you want the discriminator started', default=5)
    parser.add_argument('--save', type=str, help='save folder for backup model', default='')
    parser.add_argument('--load', type=str, help='folder backup model', default='')

    args = parser.parse_args()

    p_folder              = args.folder
    p_batch_size          = args.batch_size
    p_epochs              = args.epochs
    p_autoencoder_loss    = args.autoencoder_loss
    p_discriminator_loss  = args.discriminator_loss
    p_start_discriminator = args.start_discriminator
    p_save                = args.save
    p_load                = args.load

    if len(p_load) > 0:
        load_model = True

    if len(p_save) > 0:
        save_model = True

    # build data path
    train_path = os.path.join(p_folder, 'train')
    references_train_path = os.path.join(train_path, 'references')

    # set references as first params
    img_ref_folder = torchvision.datasets.ImageFolder(references_train_path, transform=transforms.Compose([
        #transforms.RandomVerticalFlip(1.), # flip horizontally all images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    image_folders_data = [img_ref_folder]

    # get all others data
    for feature in os.listdir(train_path):
        if feature != 'references':
            feature_train_path = os.path.join(train_path, 'references')

            img_folder = torchvision.datasets.ImageFolder(feature_train_path, transform=transforms.Compose([
                #transforms.RandomVerticalFlip(1.), # flip horizontally all images
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]))

            image_folders_data.append(img_folder)

    # shuffle data loader and made possible to keep track well of reference
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(image_folders_data),
        batch_size=p_batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
             
    train_dataset_batch_size = len(train_loader)

    # creating and loading model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ##################################
    # Autoencoder model declaration  #
    ##################################
    
    # getting input image size
    for _, data in enumerate(train_loader):
        batch_ref_data, batch_inputs = get_reference_and_inputs(data)
        n_channels = list(batch_inputs.size())[1]
        img_size = list(batch_inputs.size())[2]
        break

    # define models and loss functions
    encoder = Encoder(n_channels, img_size).to(device)
    print(encoder)

    decoder = Decoder(img_size).to(device)
    print(decoder)

    # set autoencoder parameters
    autoencoder_parameters = list(encoder.parameters()) + list(decoder.parameters())

    autoencoder_loss_func = instanciate(p_autoencoder_loss)
    autoencoder_optimizer = torch.optim.Adam(autoencoder_parameters, lr=LEARNING_RATE, betas=(0.5, 0.999))

    print('autoencoder total parameters : ', sum(p.numel() for p in autoencoder_parameters))

    ####################################
    # Discriminator model declaration  #
    ####################################
    discriminator = Discriminator(img_size).to(device)
    discriminator.apply(initialize_weights)

    discriminator_loss_func = instanciate(p_discriminator_loss)
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
        save_models_folder_path = os.path.join(BACKUP_FOLDER, p_save)

        save_global_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('global'))
        save_discriminator_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
        save_autoencoder_model_path = os.path.join(save_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    if load_model:
        load_models_folder_path = os.path.join(BACKUP_FOLDER, p_load)

        load_global_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('global'))
        load_discriminator_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('discriminator'))
        load_autoencoder_model_path = os.path.join(load_models_folder_path, BACKUP_MODEL_NAME.format('autoencoder'))

    # load models checkpoint if exists
    if load_model:

        if not os.path.exists(load_global_model_path):
            print('-------------------------')
            print('Model backup not found...')
            print('-------------------------')
        else:
            # load autoencoder state
            autoencoder_checkpoint = torch.load(load_autoencoder_model_path)

            encoder.load_state_dict(autoencoder_checkpoint['encoder_state_dict'])
            decoder.load_state_dict(autoencoder_checkpoint['decoder_state_dict'])
            autoencoder_optimizer.load_state_dict(autoencoder_checkpoint['optimizer_state_dict'])
            autoencoder_losses = autoencoder_checkpoint['autoencoder_losses']

            # load discriminator state
            discriminator_checkpoint = torch.load(load_discriminator_model_path)

            discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
            discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
            discriminator_losses = discriminator_checkpoint['discriminator_losses']

            # load global state
            global_checkpoint = torch.load(load_global_model_path)

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

         # check dataset in order to restart
        if train_dataset_batch_size * (epoch + 1) < start_iteration and restart:
            iteration += train_dataset_batch_size
            continue

        # if needed to restart, then restart from expected train_loader element
        if restart:
            nb_viewed_elements = start_iteration % train_dataset_batch_size
            indices = [ i + nb_viewed_elements for i in range(nb_viewed_elements) ]
            
            train_dataset = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_loader.dataset, indices),
                batch_size=p_batch_size, shuffle=True,
                num_workers=0, pin_memory=True)

            print('Restart using the last', len(train_dataset), 'elements of train dataset')
            restart = False
        else:
            train_dataset = train_loader

        for batch_id, data in enumerate(train_dataset):
            
            if start_iteration > iteration:
                iteration += 1
                continue
            
            # 1. get input batchs and reference

            # get reference and inputs batch data
            batch_ref_data, batch_inputs = get_reference_and_inputs(data)

            # convert batch to specific device
            batch_inputs = batch_inputs.to(device)
            batch_ref_data = batch_ref_data.to(device)
            
            # 2. Train autoencoder..
            autoencoder_optimizer.zero_grad()

            output = encoder(batch_inputs)
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
                #writer.add_image("noisy", vutils.make_grid(batch_inputs[:IMAGE_SIZE], normalize=True), iteration)
                writer.add_image("real", vutils.make_grid(batch_ref_data[:IMAGE_SIZE], normalize=True), iteration)

            # 6. Backup models information
            if iteration % BACKUP_EVERY_ITER == 0:
                if not os.path.exists(save_models_folder_path):
                    os.makedirs(save_models_folder_path)

                torch.save({
                            'iteration': iteration,
                            'encoder_state_dict': encoder.state_dict(),
                            'decoder_state_dict': decoder.state_dict(),
                            'optimizer_state_dict': autoencoder_optimizer.state_dict(),
                            'autoencoder_losses': autoencoder_losses
                        }, save_autoencoder_model_path)

                # save only if necessary (generator trained well)
                if epoch >= p_start_discriminator:
                    torch.save({
                                'model_state_dict': discriminator.state_dict(),
                                'optimizer_state_dict': discriminator_optimizer.state_dict(),
                                'discriminator_losses': discriminator_losses
                        }, save_discriminator_model_path)

                torch.save({
                            'iteration': iteration,
                            'epochs': epoch
                        }, save_global_model_path)

            # 7. increment number of iteration
            iteration += 1
        
        if epoch >= start_epoch:
            writer.add_scalar("epoch", epoch + 1, iteration)

if __name__ == "__main__":
    main()