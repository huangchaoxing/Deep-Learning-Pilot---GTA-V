# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 08:27:55 2018

@author: HP
"""
#from data_split import train_valid_split
import cv2
import torchvision
import random
from torchvision import datasets
from torchvision import transforms 
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
class DogCatdataset(Dataset):
     def __init__(self,image_dir,train):
         self.dir_list=glob.glob(image_dir+'\\*.*')
         self.dir_list=sorted(self.dir_list,key=lambda k:random.random())
         num=len(self.dir_list)
         normalization=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
         if train == 1:
             self.sub_dir_list=self.dir_list[:int(0.8*num)]
             print("train number:",len(self.sub_dir_list))
             self.trans=transforms.Compose([
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          
                                          transforms.Resize((66,200)),
                                          transforms.ToTensor(),normalization])
         else :
             self.sub_dir_list=self.dir_list[int(0.8*num):]
             self.trans=transforms.Compose([ transforms.Resize((66,200)),transforms.ToTensor(),normalization])
             print("validation number:",len(self.sub_dir_list))
     def __getitem__(self,idx):
         #print(len(self.sub_dir_list))
         image_path=self.sub_dir_list[idx]
         label=int(image_path.split("_")[-1][:-4])
         
#         if raw_label == 'cat':
#             label =1
#         else:
#             label =0
         image_data=cv2.imread(image_path)
         image_data=cv2.cvtColor(image_data,cv2.COLOR_BGR2YUV)
         image_data= Image.fromarray((image_data))
         #image_data= Image.open(image_path) 
         image_data=self.trans(image_data)
         return image_data,label
     def __len__(self):
         return len(self.sub_dir_list)
img_dir='F:\star\machine learning\ENGN8536\project\8536-PROJECT\data_collect\day3_data'


def train_valid_split():
#    normalization=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#    
#    validation_trans=transforms.Compose([transforms.ToTensor(),normalization])
#    
#    training_trans= transforms.Compose([transforms.Pad(4,0,'constant'),   #do the padding,cropping,flipping
#       transforms.CenterCrop(32),transforms.RandomHorizontalFlip(p=0.5),
#       transforms.ToTensor(),normalization])
    
    training_set=DogCatdataset(image_dir=img_dir,train=1)
    validation_set=DogCatdataset(image_dir=img_dir,train=0)
    training_loader=torch.utils.data.DataLoader(training_set,batch_size=128,num_workers=0)
    validation_loader=torch.utils.data.DataLoader(validation_set,batch_size=128,num_workers=0)
    return training_loader,validation_loader
#    train_size=int(0.8*len(whole_set))
#    test_size=len(whole_set)-train_size
#    train_dataset, test_dataset = torch.utils.data.random_split(whole_set, [train_size, test_size])
    
      
     
training_loader,validation_loader=train_valid_split()     
print(len(training_loader.dataset))
print(len(validation_loader.dataset))

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
classes=np.linspace(0,4,5)

#def imshow(img):
#   # img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
## get some random training images
#dataiter = iter(training_loader)
#image,label=dataiter.next()
#
## show images
#imshow(torchvision.utils.make_grid(image))
#print (label)
#print(classes[label])