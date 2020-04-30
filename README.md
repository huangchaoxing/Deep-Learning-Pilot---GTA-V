# ENGN8536 Final Project "END TO END LEARNING FOR SELF DRIVING PLANE IN GTA V"
## Overview
The aim of this project is to train a convolutional neural
network to automatically drive a plane in the game Grand
Theft Auto V to avoid crashing under the condition of low
altitude flight. A dataset is collected and processed and the
supervised learning method is adapted to train the CNN to
pilot the plane. We tune an architecture of a CNN in this
project and it is verified that the CNN is able to drive a
plane in GTA V .
## Outcome
[Technical Report](https://github.com/huangchaoxing/8536-project/blob/master/Final_report_chaoxing_baiyinjiya.pdf)  
[Presentation slide](https://github.com/huangchaoxing/Deep-Learning-Pilot---GTA-V/blob/master/presentation.pdf)  
[Video](https://www.youtube.com/watch?v=_yy678iEGRs&t=119s)  
[Video(for thoese who cannot access to youtube)](https://www.bilibili.com/video/av35704558)  


# Dependency
Pytorch 0.4.0  
OpenCV 3.4.1  
Grand Theft Auto Five  
# How to use:
You will need to place the GTAV window on the top left of the screen as 800*600  
Run the [data_collect.py](https://github.com/huangchaoxing/8536-project/blob/master/data_collect.py) and begin to pilot the plane(the details are explained in the technical report), the frames will be automatically labelled and saved.  
Use [little_train.py](https://github.com/huangchaoxing/8536-project/blob/master/little_train.py) to train the network.  
Use [test_model.py](https://github.com/huangchaoxing/8536-project/blob/master/test_model.py) to test the driving performance. Again, place the game window as it was during the data collection stage. Intervent the plane by using WASD when the plane is about to fail.  

# Reference
[M. Bojarski et al., ”End to end learning for selfdriving
cars,” arXiv preprint arXiv:1604.07316, 2016.](https://arxiv.org/abs/1604.07316)  
[Sentdex's py gtav](https://github.com/Sentdex/pygta5)  

