import pandas as pd
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import Dataset,SubsetRandomSampler
import torch
import matplotlib.pyplot as plt


def data_loader(train_data_location,test_data_location,batch_size,valid_size,num_workers):
    
    
    classes=['Building','Forest','Glacier','Mountain','Sea','Street']
    train_transform=transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(150),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
    test_transform=transforms.Compose([transforms.Resize(150),
                                        transforms.CenterCrop(150),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    train_data=datasets.ImageFolder(train_data_location,transform=train_transform)
    test_data=datasets.ImageFolder(test_data_location,transform=test_transform)

    num_train=len(train_data)
    indices=list(range(num_train))
    np.random.shuffle(indices)
    split=int(np.floor(valid_size*num_train)) #validation % from train_data
    train_idx,valid_idx=indices[split:],indices[:split]

    train_sampler=SubsetRandomSampler(train_idx)
    valid_sampler=SubsetRandomSampler(valid_idx)

    trainloader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,sampler=train_sampler,num_workers=num_workers)
    validloader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers)
    testloader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

    
    len_classes=len(classes)
    return len_classes,trainloader,validloader,testloader