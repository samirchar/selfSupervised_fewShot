import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import models
from torchsummary import summary
from glob import glob
import re
from itertools import compress
import pandas as pd
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(0)


class resnet18_small(nn.Module):
    def __init__(self,num_classes = 10):
        super(resnet18_small, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet18(pretrained = False)

        self.backbone.conv1 = nn.Conv2d(in_channels=self.backbone.conv1.in_channels,
                            out_channels=self.backbone.conv1.out_channels,
                            kernel_size=(3,3),
                            stride = (1,1),
                            padding=(1,1),
                            bias=False)

        self.backbone.maxpool = nn.Identity()#Remove Pooling
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features,self.num_classes) #Change output layer

    def forward(self, x):
        z = self.backbone(x)
        return z

def get_data_stats(data_loader):
  mean = 0.
  std = 0.
  nb_samples = 0.
  for data,_ in data_loader:
      batch_samples = data.size(0)
      data = data.view(batch_samples, data.size(1), -1)
      mean += data.mean(2).sum(0)
      std += data.std(2).sum(0)
      nb_samples += batch_samples

  mean /= nb_samples
  std /= nb_samples

  mean,std = list(mean.numpy().round(4)),list(std.numpy().round(4))
  return mean,std

def get_torchvision_dataset_stats(dataset_name: str, sampler = None ):
  
  data_class = getattr(torchvision.datasets,dataset_name)

  dummy_train = data_class(
      root='data',
      train=True,
      download=True,
      transform=transforms.ToTensor())

  if sampler is not None:
    #Trainset without transforms
    dummy_trainloader = torch.utils.data.DataLoader(
        dummy_train,
        batch_size=100,
        num_workers=2,
        sampler = sampler
        )
  else:
    #Trainset without transforms
    dummy_trainloader = torch.utils.data.DataLoader(
        dummy_train,
        batch_size=100,
        num_workers=2,
        )

  return get_data_stats(dummy_trainloader)

#Create Train/Val samplers

def train_val_samplers(full_train_size,val_size):

  indices = list(range(full_train_size))

  np.random.seed(43)
  np.random.shuffle(indices)

  train_idx, valid_idx = indices[val_size:], indices[:val_size]
  train_sampler = SubsetRandomSampler(train_idx)
  val_sampler = SubsetRandomSampler(valid_idx)

  return train_idx, train_sampler, valid_idx, val_sampler

def save_ckpt(epoch,
              net,
              acc,
              optimizer,
              scheduler,
              file_name = 'ckpt.pth',
              root='/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_cifar10'):
  print('Saving..')
  state = {
      'net': net.state_dict(),
      'acc': acc,
      'epoch': epoch,
      'opt': optimizer.state_dict(),
      'schd':scheduler.state_dict()
  }
  if not os.path.isdir(root):
      os.mkdir(root)
      
  torch.save(state, f'{root}/{file_name}')
  best_acc = acc

  return best_acc

def load_ckpt(net,
              optimizer,
              scheduler,
              specific_model = 'latest',
              root='/content/gdrive/My Drive/Colab Notebooks/train_checkpoints/resnet18_cifar10'):
  
  print('==> Resuming from checkpoint..')
  assert os.path.isdir(root), 'Error: no checkpoint directory found!'
  assert (specific_model == 'latest')|(specific_model == 'best')|(isinstance(specific_model,int)), f'{specific_model} not implemented'
  
  file_name = None
  
  if specific_model == 'latest':
    available_ckpts = next(os.walk(root))[-1]
    nums = [re.findall('[0-9]+',i) for i in available_ckpts]
    nums = [int(j) for i in nums for j in i if j is not None]
    latest_ckpt_number = max(nums)
    file_name = f'ckpt_epoch_{latest_ckpt_number}.pth'
    
  elif specific_model =='best':
    file_name = 'ckpt.pth'
  
  elif isinstance(specific_model,int):
    file_name = f'ckpt_epoch_{specific_model}.pth'


  checkpoint = torch.load(f'{root}/{file_name}')
  net.load_state_dict(checkpoint['net'])
  
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['opt'])

  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['schd'])

  best_acc = checkpoint['acc']
  start_epoch = checkpoint['epoch']

  return net, optimizer, scheduler,best_acc, start_epoch


def feature_extractor(model,layer_name,dataset,device,return_target = True):
  return_nodes = {layer_name:'output'}
  extractor = create_feature_extractor(model,return_nodes)
  extracted_features = []
  targets_list = []
  with torch.no_grad():
    for inputs, targets in dataset:
      inputs, targets = inputs.to(device), targets.to(device)
      features = extractor(inputs)
      extracted_features.append(features['output'].squeeze())
      targets_list.append(targets)

  extracted_features = torch.concat(extracted_features,dim=0)
  targets = torch.concat(targets_list,dim=0)
  
  if return_target:
    return extracted_features.cpu().numpy(),targets.cpu().numpy()
  
  return extracted_features.cpu().numpy()