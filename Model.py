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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Plot import *

class ImageClassificationBase(nn.Module):
    
    def __init__(self,
                    optimizer:torch.optim,
                    criterion:callable,
                    network,
                    learning_rate,
                    num_epochs
                    ) -> None:
        super(ImageClassificationBase, self).__init__()

        self.net = network
        self.criterion = criterion
        self.lr = learning_rate
        self.optimizer = optimizer(self.net.parameters(),self.lr)
        self.epochs = num_epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        

    def training_step(self, batch):
        self.net.to(self.device) 
        #print(batch[0].type)
        images, label_num = batch[0].to(self.device) ,  batch[2].to(self.device)
        out = self.net(images).to(self.device)                 # Generate predictions
        loss = self.criterion(out, label_num) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        self.net.to(self.device)

        images, label_num = batch[0].to(self.device) ,  batch[2].to(self.device)
        out = self.net(images).to(self.device)                     # Generate predictions
        loss = self.criterion(out, label_num)   # Calculate loss
        acc = Plot.accuracy(out, label_num)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def fit(self, train_loader, val_loader):
        
        history = []
        
        for epoch in range(self.epochs):
            self.net.train()
            train_losses = []

            for batch in train_loader:
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            result = Plot.evaluate(self, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)
        
        return history