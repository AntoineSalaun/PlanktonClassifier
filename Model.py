import torch.nn as nn
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

from tool_box import *

class ImageClassificationBase(nn.Module):
    
    def training_step(self, net, batch):
        images, label, label_num = batch
        out = net(images)                  # Generate predictions
        loss = F.cross_entropy(out, label_num) # Calculate loss
        return loss
    
    def validation_step(self, net, batch):
        images, labels, label_num = batch 
        out = net(images)                    # Generate predictions
        loss = F.cross_entropy(out, label_num)   # Calculate loss
        acc = Plot.accuracy(out, label_num)           # Calculate accuracy
        f1 = Plot.compute_f1(out,label_num)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'f1':f1}
        
    def validation_epoch_end(self, net, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        batch_f1 = [x['f1'] for x in outputs]
        epoch_f1 = torch.stack(batch_f1).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_f1': epoch_f1.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f},  val_f1_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['val_f1']))

    def fit(self, epochs, lr, net, train_loader, val_loader, opt_func = torch.optim.SGD):
        
        history = []
        print(net.parameters())
        optimizer = opt_func(net.parameters(),lr)
        
        for epoch in range(epochs):
            net.train()
            train_losses = []

            for batch in train_loader:
                loss = self.training_step(net, batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            result = Plot.evaluate(self, net, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)
        
        return history