from Model import ImageClassificationBase
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from torchvision.transforms import Resize
import helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

### faire une classe basenet de laquelle hérite net
## <<<<ensuite lorsque tu veux faire des nouveaux net pas be soin de recoder
## le forward et le init weight

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.apply(self.initialize_weights)

        self.network = nn.Sequential(
            
        nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
    
        nn.Flatten(),
        nn.Linear(in_features=32 * 32 * 24, out_features=84)
        #Out Features à modifier
        )


    def initialize_weights(self):
        if isinstance(self, nn.Conv2d):
            nn.init.kaiming_uniform_(self.weight.data,nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.BatchNorm2d):
            nn.init.constant_(self.weight.data, 1)
            nn.init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight.data)
            nn.init.constant_(self.bias.data, 0)


    def forward(self, xb):
        return torch.log_softmax(self.network(xb), dim=1)