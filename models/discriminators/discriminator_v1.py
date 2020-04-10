import torch

class Discriminator(torch.nn.Module):
  def __init__(self, feature_maps):
    super(Discriminator, self).__init__()
    self.feature_maps = feature_maps
    self.main = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=feature_maps, kernel_size=3, stride=1, padding=1),
                                    torch.nn.BatchNorm2d(feature_maps),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=feature_maps, out_channels=feature_maps*2, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(feature_maps * 2),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=feature_maps * 2, out_channels=feature_maps * 4, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(feature_maps * 4),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=feature_maps * 4, out_channels=feature_maps * 8, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(feature_maps * 8),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=feature_maps * 8, out_channels=feature_maps * 16, kernel_size=4, stride=1, padding=1),
                                    torch.nn.BatchNorm2d(feature_maps * 16),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=feature_maps * 16, out_channels=1, kernel_size=3, stride=2, padding=0),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim