"""
file implementing neural architecture search functionality
"""

import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

@model_wrapper
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        #add an optional convolution layer
        if nn.ValueChoice([True, False]):
            self.net.add_module(
            "layer1_conv",
            nn.Conv2d(
            in_channels=3,
            out_channels=nn.ValueChoice([16, 32, 64]), 
            kernel_size=nn.ValueChoice([3, 5, 7]), 
            stride=nn.ValueChoice([1, 2, 3]), 
            padding=nn.ValueChoice([1, 2, 3])
            )
            )
            #apply relu activation
            self.net.add_module("layer1_relu", nn.ReLU())

        #add an optional dropout layer
        if nn.ValueChoice([True, False]):
            self.net.add_module("layer1_dropout", nn.Dropout(p=nn.ValueChoice([0.1, 0.2, 0.3, 0.4, 0.5])))

        #add n fully connected layers
        for i in range(nn.ValueChoice(list(range(1,15,1)))):
            self.net.add_module(
            f"layer{i+2}_fc",
            nn.Linear(
            in_features=nn.ValueChoice([16, 32, 64, 128, 256, 512, 1024]), 
            out_features=nn.ValueChoice([16, 32, 64, 128, 256, 512, 1024])
            )
            )
            #apply relu activation
            self.net.add_module(f"layer{i+2}_relu", nn.ReLU())

            #add an optional dropout layer
            if nn.ValueChoice([True, False]):
                self.net.add_module(f"layer{i+2}_dropout", nn.Dropout(p=nn.ValueChoice([0.1, 0.2, 0.3, 0.4, 0.5])))

        #add the final output layer for binary classification
        self.net.add_module("layer_final", nn.Linear(in_features=nn.ValueChoice([16, 32, 64, 128, 256, 512, 1024]), out_features=1))

    def forward(self, x):
        x = self.net(x)
        output = torch.sigmoid(x)
        return output
