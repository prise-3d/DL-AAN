import torch
from torch import nn, optim
from torchvision import transforms, datasets

K = 64

class UShapedAutoencoder(torch.nn.Module):

    # no use of features_maps currently (set defautlt with K)
    def __init__(self, input_chanels, feature_maps):
        super(UShapedAutoencoder, self).__init__()

        # Input depth:

        # Output depth:
        # Intensity RGB
        # 3 channels

        # In 32x32
        self.encoder0 = nn.Sequential(
            nn.Conv2d(input_chanels, K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(K, K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 32x32

        # In 32x32
        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(K, 2*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2*K),
            nn.Conv2d(2*K, 2*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 16x16

        # In 16x16
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2*K, 4*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4*K),
            nn.Conv2d(4*K, 4*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 8x8

        # In 8x8 -> 4x4
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(4*K, 8*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8*K),
            nn.Conv2d(8*K, 4*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        # Out 8x8

        # In 8x8 + skip from encoder2
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(8*K, 4*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4*K),
            nn.ConvTranspose2d(4*K, 2*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Out 16x16

        # In 16x16 + skip from encoder1
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(4*K, 2*K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2*K),
            nn.ConvTranspose2d(2*K, K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Out 32x32

        # In 32x32 + skip from encoder0
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(2*K, K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(K, K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(K, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # Out 32x32

    def params(self):
        return list(self.encoder0.parameters()) + \
               list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.decoder0.parameters()) + \
               list(self.decoder1.parameters()) + \
               list(self.decoder2.parameters())


    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        x4 = self.decoder0(torch.cat((x3, x2), 1))
        x5 = self.decoder1(torch.cat((x4, x1), 1))
        x6 = self.decoder2(torch.cat((x5, x0), 1))

        return x6
