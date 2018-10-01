import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
i=0
j=1
k=2
l=3
none=4




starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    outputs=4
    #print(keys)
    
    if keys==['I']:
         outputs=i
    elif keys==['L']:
         outputs=l
    elif keys==['J']:
         outputs=j
    elif keys==['K']:
         outputs=k     
         ###############

    return outputs

key_list=[]
def main(file_name, starting_value):
    global key_list
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(8))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    frame=0
    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,40,800,600))  #set the screen size as 800*600
            last_time = time.time()
            #cv2.imshow("screen",screen)
            #cv2.waitKey(3)
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (800,600))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = keys_to_output(keys)
            #### save the training data.
            cv2.imwrite("F:\star\machine learning\ENGN8536\project\8536-PROJECT\data_collect/1_data/hcx_1_frame_"+str(frame)+'_'+str(output)+'.tif',screen)
            #training_data.append([screen,output])
            key_list.append(output)
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break
            frame+=1
            if frame % 50== 0:
                print(frame)
#                
#                if len(training_data) == 50:
#                    np.save(file_name,training_data)
#                    print('SAVED')
#                    training_data = []
#                    starting_value += 1
#                    file_name = 'training_data-{}.npy'.format(starting_value)
                    

                    
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
print(set(key_list))
import pandas as pd
result = pd.value_counts(key_list)
print (result)
day3_result=pd.DataFrame({'manipulation':result.index, 'times':result.values})

day3_result.to_csv('1_result.csv')
