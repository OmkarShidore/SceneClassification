
from torchvision import models,transforms
import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import data_loader
from net import net
import time
import numpy as np 
def train_model(model,trainloader,validloader,optimizer,loss_function,device,num_workers,num_epochs):
    since=time.time()
    print(f"Training Model")
    model.to(device)
    valid_loss_min = np.Inf
    #num_epochs=30
    #since=time.time()
    for epoch in range(num_epochs):
        train_loss=0.0
        valid_loss = 0.0
        steps=0
        model.train()
        for images,labels in trainloader:
            optimizer.zero_grad()
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*images.size(0)

        model.eval()
        for images,labels in validloader:
            images,labels=images.to(device),labels.to(device)

            outputs=model(images)
            loss=loss_function(outputs,labels)
            valid_loss+=loss.item()*images.size(0)

        train_loss=train_loss/len(trainloader.sampler)
        valid_loss=valid_loss/len(trainloader.sampler)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'train_loss':train_loss,'valid_loss':valid_loss}, 'model.pt')
            valid_loss_min = valid_loss
    time_elapsed = time.time() - since
    print('Training complete in {:.3f}m {:.3f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def test_model(model,testloader,loss_funtion,device,num_workers):
    model.to(device)
    since=time.time()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for evaluation

    for data, target in testloader:
        data,target=data.to(device),target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(testloader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(5):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    time_elapsed = time.time() - since
    print('Testing complete in {:.3f}m {:.3f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":

    #test,train folder location
    train_data_location=os.path.join(os.getcwd(),'dataset\\data\\train')
    test_data_location=os.path.join(os.getcwd(),'dataset\\data\\test\\')
    classes=['Building','Forest','Glacier','Mountain','Sea','Street']
    
    num_epochs=1
    batch_size=32
    num_workers=1
    valid_size=0.2

    #data_loader arguments=>train_loc,test_loc,
    len_classes,trainloader,validloader,testloader=data_loader(
                train_data_location,test_data_location,batch_size,valid_size,num_workers,
                )
    
    #net requires a parameter out_fetures of FC layer
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,optimizer=net(out_fetures=len_classes)
    loss_function = nn.CrossEntropyLoss()

    #train_model
    train_model(model,trainloader,validloader,optimizer,loss_function,device,1,num_epochs)

    #test_model
    test_model(model,testloader,loss_function,device,1)


    
