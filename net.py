import torch
from torch import optim
import torch.nn as nn
from torchvision import models



def net(out_fetures):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #importing AlexNet pretrained
    model=models.alexnet(pretrained=True)
    #AlexNet was orignally trained for 1000 class labels, we only have 6
    #replace final layer with a new one
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,out_fetures)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #Loss Funtion and Optimizer

    try:
        checkpoint=torch.load('model.pt')
        print('Model Found!, Loading Model')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch=checkpoint['epoch']
        train_loss=checkpoint['train_loss']
        valid_loss=checkpoint['valid_loss']
        print('Done!')
        return model,optimizer
    except:
        print('Model Not Found, Starting to train from the begining.')
        return model,optimizer