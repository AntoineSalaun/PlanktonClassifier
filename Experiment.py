import torch.nn.functional as F
from time import time


from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *


class Experiment:
    def __init__(self,
        sampling_factor = .01,
        train_factor = .8,
        num_epochs = 15,
        lr = 0.001,
        opt_func = torch.optim.Adam,
        crit = F.cross_entropy ,
        net = Net()):

        shuffle = True,
        random_seed= 42,
        batch_size = 16
        data_folder = '../ZooScanSet'
        saving_location = '../Saving_Output/'
        image_size = (128,128)
        normalize = ((0.5), (0.5))
        unwanted_classes = ['seaweed','badfocus__Copepoda','artefact','badfocus__artefact','bubble','detritus','fiber__detritus','egg__other','multiple__other']
        transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(), transforms.Normalize(*normalize)])
        dataset = PlanktonLoader(data_folder+'/taxa.csv', data_folder+"/imgs/", unwanted_classes ,transform)
        trainloader_dataset, testloader_dataset = PlanktonLoader.build_loaders(dataset, sampling_factor, train_factor, batch_size, random_seed= 41, shuffle_dataset= True)

        print('declare model')
        model = ImageClassificationBase(opt_func,crit,net,lr, num_epochs)
        print('starts training')
        history = model.fit(trainloader_dataset, testloader_dataset)
        print('starts ploting')

        exp_folder = Plot.new_folder(saving_location)
        Plot.write_param(exp_folder, sampling_factor,train_factor,num_epochs,lr,opt_func,crit)
        Plot.writ_net(exp_folder,net)
        Plot.plot_random_output(testloader_dataset, dataset, net, exp_folder+'/random_output.png')
        Plot.plot_accuracies(history, exp_folder+'/accuracy(e).png')
        Plot.plot_losses(history, exp_folder+'/losses.png')
        df = Plot.class_accuracies(Plot, net, dataset, testloader_dataset, exp_folder+'/class_acc.csv')
