import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def single_conv5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5),
        nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
        nn.Tanh()
    )
    
def single_conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
        nn.Tanh()
    )
    
def single_conv2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 2),
        nn.BatchNorm2d(out_channels, eps = 1e-5, momentum=0.1),
        nn.Tanh()
    )

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, Cin, action_size, out_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.maxpool = nn.MaxPool2d(2)

        self.layer1 = single_conv3(Cin, 16)
        self.layer2_1 = single_conv5(16, 32)
        self.layer2_2 = single_conv5(32, 32)
        #self.layer2 = single_conv2(16, 16)
        self.layer2_3 = single_conv3(32, 32)
        self.layer3 = single_conv3(32, 64)
        self.layer4 = single_conv5(64, 128)
        
        self.fc1 = nn.Sequential(
                nn.Linear(128*3*3+action_size, 64),
                nn.ReLU(inplace=True)
                )
        self.fc2 = nn.Linear(64, out_size)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        """
        Action based on state
        :param state1: -1, 9, 21, 21
        :param state2: -1, 21
        :return: action1, action2
        """
        x1 = self.layer1(x1)                                 ### -1, 16, 19, 19
        #print(f'conv1: {conv1.data.cpu().numpy()[0,0,:,:]}')
        
        x1 = self.layer2_1(x1)                                 ### -1, 32, 15, 15
        #print(f'conv2: {conv2.data.cpu().numpy()[0,0,:,:]}')
        
        x1 = self.layer2_2(x1)                                ### -1, 32, 11, 11
        
        x1 = self.layer2_3(x1)                                ### -1, 32, 9, 9
        
        #x1 = self.maxpool(x1)                                ### -1, 16, 9, 9           
        
        x1 = self.layer3(x1)                                 ### -1, 64, 7, 7
        #print(f'conv3: {conv3.data.cpu().numpy()[0,0,:,:]}')
        
        x1 = self.layer4(x1)                                 ### -1, 128, 3, 3
        #print(f'conv4: {conv4.data.cpu().numpy()[0,0,:,:]}')
        
        x1 = x1.view(-1, 128*3*3)                             ### -1, 128*3*3
        
        x = torch.cat([x1, x2], dim=1)                       ### -1, 128*3*3+21          

        x = self.fc1(x)                                      ### -1, 64
        #print(f'lconv6: {left_conv6.data.cpu().numpy()[0,0,:,:]}')
        
        out = self.fc2(x)                                      ### -1, 8

        return out
