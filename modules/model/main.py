import torch
import torch.nn as nn


class ConvCatNet(nn.Module):
    def __init__(self, image_size=28):
        super(ConvCatNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0)

        # First fully connected layer
        self.fc1 = nn.Linear(324, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = torch.flatten(self.conv1(x), 1)
        # x2 = torch.flatten(self.conv2(x), 1)

        # x = torch.cat([x1, x2], dim=1)
        x = x1
        x = self.fc1(x)
        x = self.lrelu(x)

        x = self.fc2(x)

        output = torch.sigmoid(x)
        return output
