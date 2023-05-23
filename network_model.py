
"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.  
 """

import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models


class TunedResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048,512),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.ELU(),
        )
    def forward(self, input):
        input = self.resnet50(input)
        return input
    
    def get_fc_layers(self,):
        return self.resnet50.fc.parameters()
    
    def get_main_layers(self,):
        return [param for name, param in self.resnet50.named_parameters() if 'fc' not in name]
    
class NvidiaModel(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self):
        super().__init__()
        self.nvidia = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
        )
    """ 
    * @brief Function to build the model.
    * @parma The image to train.
    * @return The trained prediction network.
    """
    def forward(self, input):
        input = self.nvidia(input)
        return input