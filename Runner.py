import torch 
from torchvision import transforms
import torch.nn.functional as F

from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *
from Experiment import *


##############################################



exp1 = Experiment(
sampling_factor = .1,
train_factor = .8,
num_epochs = 2,
lr = 0.001,
opt_func = torch.optim.Adam,
crit = F.cross_entropy ,
net = Net())

exp2 = Experiment(
sampling_factor = .001,
train_factor = .7,
num_epochs = 6,
lr = 0.002,
opt_func = torch.optim.Adam,
crit = F.cross_entropy ,
net = Net())


