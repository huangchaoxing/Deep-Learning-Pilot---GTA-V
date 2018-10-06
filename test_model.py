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




def dive():
 PressKey(I)
 ReleaseKey(J)
 ReleaseKey(K)
 ReleaseKey(L)
 print("dive")
 
def pull_up():
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
    PressKey(L)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(J)   
    print("right")
    
def release():
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(J)  
    print("release")
### The code  being commented below is used to shown the output of a single input image
## you can use the function imshow(img) to check if the normalization works fine(if the img in the evening is totally dark,then it is not good)    
model=torch.load("37500_net.pkl")    
#image=cv2.imread("day3_data/hcx_1_frame_16521_2.tif")
normalization=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
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
   
#output=model(data)
#print(output)

def main():
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

        
            if 'T' in keys:
                duration=time.time()-last_time
                print("last for",duration,"seconds")
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    release()
                    time.sleep(1)    
                    
main()                    



 
 
