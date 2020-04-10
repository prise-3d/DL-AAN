import torch

class AutoEncoder(torch.nn.Module):
    
    def __init__(self, input_chanels, feature_maps):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_chanels, feature_maps)
        self.decoder = Decoder(feature_maps)

    def params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def forward(self, inp):
        return self.decoder(self.encoder(inp))
    

class Encoder(torch.nn.Module):
  def __init__(self, input_chanels, feature_maps):
    super(Encoder, self).__init__()
    self.encoder = torch.nn.Sequential(
                                       torch.nn.Conv2d(input_chanels, feature_maps, kernel_size=3, stride=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(feature_maps),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(feature_maps * 2),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(feature_maps * 4),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=3, stride=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(feature_maps * 8),
                                       torch.nn.Dropout(0.3),
                                       torch.nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=3, stride=1),
                                       torch.nn.LeakyReLU(0.2, inplace=True),
                                       torch.nn.BatchNorm2d(feature_maps * 16),
                                       torch.nn.Dropout(0.3),
                                       #Flatten()
                                      )
  def forward(self, inp):
    return self.encoder(inp)


class Decoder(torch.nn.Module):
  def __init__(self, feature_maps):
    super(Decoder, self).__init__()
    self.decoder = torch.nn.Sequential(
                                    #UnFlatten(),
                                    torch.nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, kernel_size=3, stride=1),
                                    torch.nn.LeakyReLU(),
                                    #torch.nn.BatchNorm2d(feature_maps * 8),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=3, stride=1),
                                    torch.nn.LeakyReLU(),
                                    #torch.nn.BatchNorm2d(feature_maps * 4),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=3, stride=1),
                                    torch.nn.LeakyReLU(),
                                    #torch.nn.BatchNorm2d(feature_maps * 2),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=3, stride=1),
                                    torch.nn.LeakyReLU(),
                                    #torch.nn.BatchNorm2d(feature_maps),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.ConvTranspose2d(feature_maps, 3, kernel_size=3, stride=1),
                                    torch.nn.ReLU(),
                                   )
  def forward(self, inp):
    return self.decoder(inp)