from torch.utils.data import Dataset, DataLoader, random_split
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
import os
import datetime


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
        plt.savefig(saving_location)
        plt.close()


    def accuracy(outputs, labels): 
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    
    

    def plot_accuracies(history, num_epochs, saving_location_g,saving_location_d):
        """ Plot the history of accuracies"""
        accuracies = [x['val_acc'] for x in history]
        pd.DataFrame(np.column_stack([ accuracies]), columns=['accuracies']).to_csv(saving_location_d) 
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.savefig(saving_location_g)
        plt.close()

    def plot_losses(history,num_epochs,saving_location_g, saving_location_d):
        """ Plot the losses in each epoch"""
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        pd.DataFrame(np.column_stack([ train_losses,val_losses ]), columns=['train loss', 'val loss']).to_csv(saving_location_d) 

        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.savefig(saving_location_g)
        plt.close()

    def plot_random_output(testloader_dataset, dataset, model, saving_location):
        
        images, labels, label_num = next(iter(testloader_dataset)).to(model.device)
        fig=plt.figure(figsize=(15, 15))
        _, preds = torch.max(model.net(images).cpu(), dim=1)
        
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
        plt.close()

        print(count, 'good predictions. Accuracy : ', count/len(preds) )


    def class_success(outputs, label_num):
        _, preds = torch.max(outputs.cpu(), dim=1)
        tries, successes = np.zeros(84), np.zeros(84)

        for i in range(len(preds)):
            tries[label_num] = tries[label_num] + 1
            if preds[i] == label_num[i]:
                successes[label_num] = successes[label_num] + 1
        return successes, tries

    def class_accuracies(self, model, dataset, val_loader, saving_location):
        pred_per_class, correct_pred_per_class = np.zeros(84), np.zeros(84)
        accuracy_per_class = np.zeros(len(pred_per_class))

        for batch in val_loader:
            successes, passes = self.class_success(model.net(batch[0].to(model.device)).cpu() , batch[2].cpu())
            correct_pred_per_class, pred_per_class = correct_pred_per_class + successes, pred_per_class + passes    

        pd.DataFrame(np.column_stack([dataset.classes, pred_per_class, accuracy_per_class]), columns=['Class', '# predictions ', 'Accuracy']).to_csv(saving_location)

        #for i in range(len(pred_per_class)):
        #    if pred_per_class[i] == 0:
        #        print('Class #', i, ' ', dataset.classes[i] , 'was never trained on')
        #    else:
        #        accuracy_per_class[i] = correct_pred_per_class[i] / pred_per_class[i]
        #        print('Class #', i, ' ', dataset.classes[i] , 'trained on', int(pred_per_class[i]) ,'times -> accuracy :', accuracy_per_class[i])

    

    def new_folder(saving_location):
        folder_name = saving_location+'Experiment_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(folder_name)
        return folder_name
    
    def write_param(exp_name, wd, batch_size, sampling_factor,train_factor,num_epochs,lr,opt_func,crit):
        with open(exp_name+'/parameters.txt', 'w') as f:
            f.write("sampling_factor = " + str(sampling_factor) + "\n" 
            + "train_set_proportion = " + str(train_factor) + "\n" 
            + "num_epochs = " + str(num_epochs) + "\n" 
            + "learning_rate = " + str(lr) + "\n" 
            + "optimizing_function = " + str(opt_func) + "\n" 
            + "Loss = " + str(crit) + "\n" 
            + "batch_size = " + str(batch_size) + "\n"
            + "weight_decay = " + str(wd) + "\n")

    def writ_net(exp_folder,net):
        with open(exp_folder+'/network_architecture.txt', 'w') as f:
            f.write(str(net))

    def export_results(model, wd, batch_size, net, history, dataset, testloader_dataset, saving_location, sampling_factor,train_factor,num_epochs,lr,opt_func,crit): 
            exp_folder = Plot.new_folder(saving_location)
            Plot.write_param(exp_folder, wd, batch_size, sampling_factor,train_factor,num_epochs,lr,opt_func,crit)
            Plot.writ_net(exp_folder,net)
            Plot.plot_accuracies(history, num_epochs, exp_folder+'/accuracy(e).png', exp_folder+'/accuracy(e).csv')
            Plot.plot_losses(history, num_epochs, exp_folder+'/losses.png', exp_folder+'/losses(e).csv')
            Plot.class_accuracies(Plot, model, dataset, testloader_dataset, exp_folder+'/class_acc.csv')