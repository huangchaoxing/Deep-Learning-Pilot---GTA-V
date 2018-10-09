# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:45:04 2018

@author: HP
"""
from nvidia_data_gathering import train_valid_split
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import time
ITERATE_NUM=15
LR=1e-4

#preload=models.resnet18(pretrained=True)
#for parameters in preload.parameters():
#    parameters.requires_grad=False
  
#features_number=preload.fc.in_features
#preload.fc=nn.Linear(features_number,5)
#preload.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
           #                    bias=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


    
    
 
    
    
class nvidia(nn.Module):    
       def __init__(self):
           super(nvidia, self).__init__()
           self.BN=nn.BatchNorm2d(3)
           self.conv1=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=5,stride=2)
           self.relu=nn.ReLU()
           self.conv2=nn.Conv2d(in_channels=24,out_channels=36,kernel_size=5,stride=2)
           self.conv3=nn.Conv2d(in_channels=36,out_channels=48,kernel_size=5,stride=2)
           self.conv4=nn.Conv2d(in_channels=48,out_channels=64,kernel_size=3,stride=1)
           self.conv5=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
           self.FC1=nn.Linear(64*18,1164)
           self.FC2=nn.Linear(1164,100)
           self.FC3=nn.Linear(100,50)
           self.FC4=nn.Linear(50,10)
           self.output=nn.Linear(10,5)
           
       def forward(self,x):
           x=self.BN(x)
           x=self.conv1(x)
           #print(x.shape)
           x=self.relu(x)
           x=self.conv2(x)
           x=self.relu(x)
           x=self.conv3(x)
           x=self.relu(x)
           x=self.conv4(x)
           x=self.relu(x)
           x=self.conv5(x)
           x=self.relu(x)
           x=x.view(x.size(0),-1)
           x=self.FC1(x)
           x=self.relu(x)
           x=self.FC2(x)
           x=self.relu(x)
           #print(x.shape)
           x=self.FC3(x)
           x=self.relu(x)
           x=self.FC4(x)
           x=self.relu(x)
           x=self.output(x)
           return x
           
           
           

#model=nvidia()
model=model.to(device)

criterion=nn.CrossEntropyLoss()

#optimizer=optim.Adam([{'params':model.parameters()}],lr=LR,betas=(0.9,0.999))
optimizer=optim.Adam( model.parameters() ,lr=LR,betas=(0.9,0.999),weight_decay=0)

    
train_loss_result={}
epoch_valid_acc={}
epoch_train_acc={}
epoch_valid_loss_result={}

training_loader,validation_loader=train_valid_split()
step=0
print("Ready !")
start_time = time.time()
for epoch in range(ITERATE_NUM):  # loop over the dataset multiple times
    epoch_start_time=time.time()
    epoch_loss=0
    epoch_valid_loss=0
    running_loss = 0.0
    print("This is epoch",epoch)
    duration=0
    batch_num=0
   # step_per_epoch=int(30494/16)
    for i, data in enumerate(training_loader, 0):
        batch_start_time=time.time()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()                        # let's do the forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()                              # let's backprop
        optimizer.step()
        epoch_loss +=loss.item()
        
        batch_num=batch_num+1           
        running_loss = 0.0
        batch_end_time=time.time()
        duration=duration+batch_end_time-batch_start_time
        step=step+1
       # print("step:",step,'/',step_per_epoch,'loss',loss.item())
        
        if step%10==0:
            print("loss, ",loss.item(),"step,",step)
            #print("label,",labels)
            _, prediction = torch.max(outputs.data,1)
            #print("outputs,",prediction)
        if step%200==0:
            torch.save(model,str(step)+'_Nvidia_net.pkl')
    print("avg duration is",duration/batch_num) 
    print("epoch time is",time.time()-epoch_start_time)
    epoch_loss=epoch_loss/(20002/128)
    train_loss_result[epoch]=epoch_loss
    print ("The epoch loss is",epoch_loss) 
    
    right=0
    right_train=0
    

    with torch.no_grad():  
      total_valid=0  
      for j,data in enumerate (validation_loader):  
         
          images,labels=data
          images, labels = images.to(device), labels.to(device)
          valid_output=model(images)
          _, prediction = torch.max(valid_output.data,1)
          right = right+(prediction == labels).sum().item()
    valid_accuracy=right/5001
    epoch_valid_acc[epoch]=valid_accuracy      
    print("The epoch validation accuracy is",valid_accuracy)

    if epoch%5==0 or epoch==29:
        with torch.no_grad():
            for data in training_loader:
               
                images,labels=data
                images, labels = images.to(device), labels.to(device)
                training_output=model(images)
                _, prediction = torch.max(training_output.data,1)
                right_train= right_train+(prediction == labels).sum().item()
        training_accuracy=right_train/20002
        epoch_train_acc[epoch]=training_accuracy      
        print("The epoch training accuracy is",training_accuracy)
    

    with torch.no_grad():
     for i,data in enumerate(validation_loader):
        inputs,label=data
        inputs, label = inputs.to(device), label.to(device)
        outputs=model(inputs)
        valid_loss=criterion(outputs,label)
        epoch_valid_loss += valid_loss.item()
    epoch_valid_loss=epoch_valid_loss/(5001/128)
    epoch_valid_loss_result[epoch]=epoch_valid_loss
    print("The epoch valid loss is",epoch_valid_loss)
    print("******************************")     

print("training loss vs epoch",train_loss_result) 
print("valid acc vs epoch",epoch_valid_acc)  
print("training acc vs epoch",epoch_train_acc) 
print("valid loss vs epoch",epoch_valid_loss_result) 
print('Finished Training')

epoch_value=list(epoch_valid_acc.keys())
epoch_array= np.array(epoch_value)
train_loss_array=np.array(list(train_loss_result.values()))
train_acc_array=np.array(list(epoch_train_acc.values()))
valid_acc_array=np.array(list(epoch_valid_acc.values()))
valid_loss_array=np.array(list(epoch_valid_loss_result.values()))
mean_valid_acc=np.mean(valid_acc_array)
print("mean_valid_acc of this model is",mean_valid_acc)
mean_train_acc=np.mean(train_acc_array)
print("mean_train_acc of this model is",mean_train_acc)

# Training finish, lets do the plotting 

plt.figure(1)
plt.plot(epoch_array,train_loss_array)
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.title('Training loss vs epochs')
plt.show()
#plt.savefig("1")

epoch_train_acc_array=np.array(list(epoch_train_acc.keys()))
plt.figure(2)
plt.plot(epoch_train_acc_array,train_acc_array)
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.title('Training accuracy vs epochs')
plt.show()
#plt.savefig("2")

plt.figure(3)
plt.plot(epoch_array,valid_acc_array)
plt.xlabel('epochs')
plt.ylabel('validation accuracy')
plt.title('validation accuracy vs epochs')
plt.show()
#plt.savefig("3")

plt.figure(4)
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation loss vs epochs')
plt.plot(epoch_array,valid_loss_array)
plt.show()
print('total time',time.time()-start_time)    

