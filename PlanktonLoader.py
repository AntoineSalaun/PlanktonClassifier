import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import RandomSampler, DataLoader, Subset, SubsetRandomSampler, RandomSampler


class PlanktonLoader(Dataset):
    """Loads the plankton Classification dataset."""

    def __init__(self, csv_file, image_folder, unwanted_classes = None, transform=None):
        #Trying to delete unwanted files
        try:
            for i in unwanted_classes:  
                shutil.rmtree(image_folder+i)
        except (FileNotFoundError):
            pass

        self.data_pre = pd.read_csv(csv_file)
        self.data = self.data_pre[~self.data_pre.taxon.str.contains('|'.join(unwanted_classes))]
        #réindexer
        self.data.index = range(len(self.data))
        self.transform = transform

        # First 2 columns contains the id for the image and the class of the image
        self.dict = self.data.iloc[:,:2].to_dict()
        # When we index we want to get the id
        self.ids = self.dict["objid"]
        print(' The id list has a lenght of ', len(self.ids))
        self.classes = self.data["taxon"].unique() # List of unique class name
        # Comparer classes of self.data and the folders on the computer
      
        if set(os.listdir(image_folder)) == set(self.classes) :
            print('Folder list corresponds to classes of interes')
        else:
            print('oops identation broken, the following differs between the list of directories and the class list')
            print(list(set(self.classes) - set(os.listdir(image_folder)) ))
            print(list(set(os.listdir(image_folder))-set(self.classes) ))



        self.class_to_idx = {j: i for i, j in enumerate(self.classes)} 
        # Assigns number to every class in the order which it appears in the data
        self.species = self.dict["taxon"]
        # Use this go back to class name from index of the class name
        self.path_plankton = image_folder # Where the images are stored

        print('We have ', len(self.classes), 'classes')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()
            assert isinstance(idx, int)

        num = self.ids[idx] # Id of the indexed item
        loc = f"/{num}.jpg"
        label = self.dict["taxon"][idx] # Find the label/class of the image at given index
        label_num = self.class_to_idx[label] # Convert it to int
        image = Image.open(self.path_plankton + self.dict["taxon"][idx] + loc)
        if self.transform:
            image = self.transform(image)

        return (image, label, label_num)

    def build_loaders(dataset, sampling_factor, train_factor, batch_size = 16, random_seed= 42 , shuffle_dataset = True  ): 
        num_samples = int(sampling_factor * len(dataset))
        train_size = int(train_factor * num_samples) # train_factor % of the data to be used for training
        test_size = num_samples - train_size # The remainder for testing

        print('We use ', sampling_factor, 'of the data (',num_samples, 'samples) and the train factor is ', train_factor)
        print('Train set contains', train_size, 'images.')
        print('Test set contains', test_size, 'images.')


        train_dataset, test_dataset = random_split(Subset(dataset, np.random.permutation(np.arange(int(len(dataset))))[:train_size+test_size]), [train_size, test_size])

        # put in batches :
        trainloader_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return trainloader_dataset, testloader_dataset