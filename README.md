# 8536-project
The data_gathering.py is used for dataloading in pytorch. The little_train.py contains a small model for a trial train. We use data_collect.py to get data(remember to include the grabscreens.py and getkeys.py )

For the Nvidia archetiecture, codes are similar. We just resize the image as 66*200. And before feeding the data into the network, we first convert the color to YUV channel. So if you want to test the Nvidia model, remember to convert the image into YUV channel before testing. 

For the test_model.py file, the test metric is autonomy(please refer to https://arxiv.org/abs/1604.07316 ). When test the model, the program will pause after 120s, thus we are testing the autonomy within 120s. Remember within that 120s, you can use WASD to prevent the plane from crashing(WASD is modified into control rolling and pitch and we use up down left right arrow to control yaw and throttle). Everytime that you think the CNN is going to crash the plane, you can use the WASD to save the plane. After 120s, the code will count how many times you save the plane and use this data to caculate the autonomy. Repeat the 120s test for 5 times and calucualte  the average autonomy.

