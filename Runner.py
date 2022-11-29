from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
from torchmetrics import F1Score
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
import os
from torch.utils.data import RandomSampler, DataLoader, Subset, SubsetRandomSampler, RandomSampler


from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *


##############################################

sampling_factor = .01
train_factor = .8
shuffle = True
random_seed= 42
num_epochs = 8
lr = 0.001
opt_func = torch.optim.Adam

#############################################

batch_size = 16
data_folder = '../ZooScanSet'
saving_location = '../Saving_Output/'
image_size = (128,128)
normalize = ((0.5), (0.5))
unwanted_classes = ['seaweed','badfocus__Copepoda','artefact','badfocus__artefact','bubble','detritus','fiber__detritus','egg__other','multiple__other']
transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(), transforms.Normalize(*normalize)])
dataset = PlanktonLoader(data_folder+'/taxa.csv', data_folder+"/imgs/", unwanted_classes ,transform)
trainloader_dataset, testloader_dataset = PlanktonLoader.build_loaders(dataset, sampling_factor, train_factor, batch_size, random_seed= 41, shuffle_dataset= True)

###############################################

net = Net()
net.initialize_weights()
model = ImageClassificationBase()
history = ImageClassificationBase.fit(model, num_epochs, lr, net, trainloader_dataset, testloader_dataset, opt_func)

###############################################

torch.save(net.state_dict(), saving_location+'/last_model.pth')
Plot.plot_random_output(testloader_dataset, dataset, net, saving_location+'random_output.png')
Plot.plot_accuracies(history, saving_location+'accuracy(e).png')
Plot.plot_losses(history, saving_location+'losses.png')
df = Plot.class_accuracies(Plot, net, dataset, testloader_dataset, saving_location+'class_acc.csv')



