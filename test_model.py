# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:07:07 2018

@author: HP
"""
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, I, J, K, L
from matplotlib import pyplot as plt
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms 
from PIL import Image

global count
global interfere
count=0
interfere=0
###put your network class here to ensure the model loading can recognize the model.

###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv0=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2),
#                                 nn.ReLU(), 
#                                 nn.BatchNorm2d(16))
                                # nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(32))                       
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(64),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(128),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(256),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(256),
                                 nn.MaxPool2d(kernel_size=2,stride=1))
        
        self.last=nn.Sequential(nn.Linear(23*23*256,512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.Linear(256,5))

    def forward(self, x):
        #x=self.conv0(x)
        
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        #print(x.shape)
        x=x.view(x.size(0),-1)
        
        output= self.last(x)
        return output    

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


def dive():
 global count
 count+=1   
 PressKey(I)
 ReleaseKey(J)
 ReleaseKey(K)
 ReleaseKey(L)
 print("dive")
 
def pull_up():
    global count
    count+=1
    PressKey(K)
    ReleaseKey(J)
    ReleaseKey(I)
    ReleaseKey(L)
    print("pull up")
    
def left():
    PressKey(J)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(L)  
    print("left")
    
def right():
    global count
    count+=1
    PressKey(L)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(J)   
    print("right")
    
def release():
    global count
    count+=1
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(J)  
    print("release")
### The code  being commented below is used to shown the output of a single input image
## you can use the function imshow(img) to check if the normalization works fine(if the img in the evening is totally dark,then it is not good)    
model=torch.load("small_net/37500_net.pkl")    
#image=cv2.imread("day3_data/hcx_1_frame_13975_1.tif")
normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
trans=transforms.Compose([ transforms.Resize((100,100)),transforms.ToTensor(),normalization])
#data = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
#def imshow(img):
#   
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
#data=trans(data)    
#imshow(torchvision.utils.make_grid(data)) 
#data=data.unsqueeze(0)
device= torch.device("cuda:0")
#data=data.to(device)
#   
#output=model(data)
#print(output)
#time.sleep(60)

def main():
    global interfere
    print("ready")
    for i in list(range(8))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()
    paused = False    
    while(True):
        if not paused:
            screen = grab_screen(region=(0,40,800,600))
            screen=cv2.resize(screen,(800,600))
            #screen=cv2.cvtColor(screen,cv2.COLOR_BGR2YUV)
            data = Image.fromarray((screen))
            data=trans(data)
            data=data.unsqueeze(0)
            data=data.to(device)
            output=model(data)
            _, prediction = torch.max(output.data,1)
            if prediction ==0:
                dive()
            if prediction==1:
                left()
            if prediction==2:
                pull_up()
            if prediction==3:
                right()
            if prediction==4:
                release()
            keys = key_check()
            duration=time.time()-last_time
            if duration>120:
                automatic=1-(interfere/count)
                print("last for",duration,"seconds")
                print("automatic",automatic)
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    release()
                    time.sleep(1) 
                    
            if 'W' in keys or 'A' in keys or 'S' in keys or 'D' in keys:
                interfere+=1
            if 'T' in keys:
                duration=time.time()-last_time
                automatic=1-(interfere/count)
                print("last for",duration,"seconds")
                print("automatic",automatic)
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    release()
                    time.sleep(1)    
                    
main()                    



 
 
