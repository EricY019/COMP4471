import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import datetime
import numpy as np
import joint_transforms
from dataset import CaptchaFolder


model_save_path = ''
exp_name = ''
path = ''
label_path = ''

args = {
    'epoch': 10,
    'train_batch_size': 5,
    'seed': 2021,
    'img_size_h': 100,
    'img_size_w': 100,
}

# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# cudnn settings
cudnn.deterministic = True
cudnn.benchmark = True

# single-GPU training
torch.cuda.set_device(0)
batch_size = args['train_batch_size']

# transforms
transform = transforms.ToTensor()

# load dataset
print('=====>Loading dataset<======')
train_set = CaptchaFolder(img_path=path, label_path=label_path, transform=transform, batch_size=args['train_batch_size'], split='TRAIN')
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)

val_set = CaptchaFolder(img_path=path, label_path=label_path, transform=transform, batch_size=args['train_batch_size'], split='VAL')
val_loader = DataLoader(val_set, batch_size=args['train_batch_size'], num_workers=8)

#log_path = os.path.join(model_save_path, exp_name, str(datetime.datetime.now()) + '.txt')
#log_path_val = os.path.join(model_save_path, exp_name, str(datetime.datetime.now()) + '_val.txt')

# loss?

def main():
    
    net = 
    net.cuda().train()
    
    train_params = []
    
    optimizer = 