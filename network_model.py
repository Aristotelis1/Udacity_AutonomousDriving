
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
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000,512),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            # nn.ELU(),
            )
        # Freeze the first 45 layers
        n = 45  # Specify the number of layers to freeze

        for idx, (name, param) in enumerate(self.resnet50.named_parameters()):
            if idx < n:
                param.requires_grad = False
            else:
                break

    def forward(self, input):
        input = self.resnet50(input)
        input = self.fc(input)
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


class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output