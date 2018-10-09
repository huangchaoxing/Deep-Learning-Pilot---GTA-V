# 8536-project
The data_gathering.py is used for dataloading in pytorch. The little_train.py contains a small model for a trial train. We use data_collect.py to get data(remember to include the grabscreens.py and getkeys.py )

For the Nvidia archetiecture, codes are similar. We just resize the image as 66*200. And before feeding the data into the network, we first convert the color to YUV channel. So if you want to test the Nvidia model, remember to convert the image into YUV channel before testing. 
