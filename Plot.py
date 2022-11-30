from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
import torchvision
from torchmetrics import F1Score
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
from datetime import datetime

class Plot():
    def subplot_random(trainloader_dataset, saving_location):
        im, lab, lab_num = next(iter(trainloader_dataset))
        fig=plt.figure(figsize=(15, 15))

        for idx,(i,j) in enumerate(zip(im,lab)):
            print('hop')
            ax = fig.add_subplot(4,4,idx)
            print(i.squeeze().numpy())
            ax.imshow(i.squeeze().numpy())       
            ax.set_title(j)
            #ax.set_title()
        plt.show()
        plt.savefig(saving_location)

    def accuracy(outputs, labels): 
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def plot_accuracies(history, saving_location):
        """ Plot the history of accuracies"""
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.savefig(saving_location)

    
    def plot_f1(history, saving_location):
        """ Plot the history of accuracies"""
        f1 = [x['val_f1'] for x in history]
        plt.plot(f1, '-x')
        plt.xlabel('epoch')
        plt.ylabel('f1-score')
        plt.title('f1-socre vs. No. of epochs')
        plt.savefig(saving_location)

    # For a given batch, it should return twp arrays of length 84 corressponding to the number of successful classification per class
    # and the number of tries per class

    def plot_losses(history,saving_location):
        """ Plot the losses in each epoch"""
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.savefig(saving_location)


    def plot_random_output(testloader_dataset, dataset, net, saving_location):
        images, labels, label_num = next(iter(testloader_dataset))
        fig=plt.figure(figsize=(15, 15))
        _, preds = torch.max(net(images), dim=1)
        
        count = 0
        for idx,(i,j,k) in enumerate(zip(images, labels, preds)):
            idx += 1
            ax = fig.add_subplot(4,4,idx)
            ax.imshow(i.squeeze().numpy())          
            title = j + ' -> ' + dataset.classes[k]
            ax.set_title(title)
            if j == dataset.classes[ k] :
                count = count +1
        plt.savefig(saving_location)
        plt.show()

        print(count, 'good predictions. Accuracy : ', count/len(preds) )


    def class_success(outputs, label_num):
        _, preds = torch.max(outputs, dim=1)
        tries = np.zeros(84)
        successes = np.zeros(84)
        for i in range(len(preds)):
            tries[label_num] = tries[label_num] + 1
            if preds[i] == label_num[i]:
                successes[label_num] = successes[label_num] + 1
        return successes, tries

    def class_accuracies(self, net, dataset, val_loader, saving_location):
        pred_per_class = np.zeros(84)
        correct_pred_per_class = np.zeros(84)
        accuracy_per_class = np.zeros(len(pred_per_class))

        for batch in val_loader:
            images, labels, label_num = batch
            out = net(images)
            successes, passes = self.class_success(out, label_num)
            correct_pred_per_class = correct_pred_per_class + successes
            pred_per_class = pred_per_class + passes

        for i in range(len(pred_per_class)):
            if pred_per_class[i] == 0:
                print('Class #', i, ' ', dataset.classes[i] , 'was never trained on')
            else:
                accuracy_per_class[i] = correct_pred_per_class[i] / pred_per_class[i]
                print('Class #', i, ' ', dataset.classes[i] , 'trained on', int(pred_per_class[i]) ,'times -> accuracy :', accuracy_per_class[i])

        df = pd.DataFrame(np.column_stack([dataset.classes, pred_per_class, accuracy_per_class]), columns=['Class', '# predictions ', 'Accuracy'])
        df.to_csv(saving_location)
        return df

    def compute_f1(preds,target):
        f1 = F1Score(num_classes=84)
        return f1(preds, target)

    def new_folder(saving_location):
        date_time_string = datetime.now.strftime("%m/%d/%Y_%H:%M:%S")
        folder_name = saving_location+'Experiment_'+ date_time_string
        os.makedirs(folder_name, exist=True)
        return folder_name
    
    def write_param(exp_name, sampling_factor,train_factor,num_epochs,lr,opt_func,crit):
        with open(exp_name+'/parameters.txt', 'w') as f:
            f.write("sampling_factor = " + sampling_factor + "\n" 
            + "train_set_proportion = " + train_factor + "\n" 
            + "num_epochs = " + num_epochs + "\n" 
            + "learning_rate = " + lr + "\n" 
            + "optimizing_function = " + opt_func + "\n" 
            + "Loss = " + crit + "\n" )

    def writ_net(exp_folder,net):
        with open(exp_folder+'/network_architecture.txt', 'w') as f:
            f.write(net)