
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

"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
class model_cnn(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
    """ 
    * @brief Function to build the model.
    * @parma The image to train.
    * @return The trained prediction network.
    """
    def forward(self, input):
        input = input/127.5-1.0
        input = self.elu(self.conv_0(input))
        input = self.elu(self.conv_1(input))
        input = self.elu(self.conv_2(input))
        input = self.elu(self.conv_3(input))
        input = self.elu(self.conv_4(input))
        input = self.dropout(input)

        input = input.flatten()
        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        input = self.elu(self.fc2(input))
        input = self.fc3(input)

        return input

class TunedResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048,100),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 10),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 1),
            nn.ELU(),
        )
    def forward(self, input):
        input = self.resnet50(input)
        return input
    
    def get_fc_layers(self,):
        return self.resnet50.fc.parameters()
    
    def get_main_layers(self,):
        return [param for name, param in self.resnet50.named_parameters() if 'fc' not in name]
        