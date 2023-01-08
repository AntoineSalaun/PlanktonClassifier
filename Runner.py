import torch 
from torchvision import transforms
import torch.nn.functional as F

from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *
from Experiment import *

############################################## 


exp16 = Experiment(
sampling_factor = 1,
train_factor = .7,
num_epochs = 20,
lr = 0.0001,
batch_size = 32,
opt_func = torch.optim.SGD,
crit = F.cross_entropy ,
net = DFL_VGG16())