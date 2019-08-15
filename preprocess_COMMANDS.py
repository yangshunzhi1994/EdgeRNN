import re
import os
import math
import librosa
import pickle
import h5py
import scipy
import pandas as pd
from tqdm import tqdm
import numpy as np 
import glob

train_test_scale = 0.8

len_Yes = 2377
len_No = 2367
len_Up = 2375
len_Down = 2359
len_Left = 2353
len_Right = 2367
len_On = 2375
len_Off = 2357
len_Stop = 2380
len_Go = 2372

len_train1 = [int(len_Yes*train_test_scale), int(len_No*train_test_scale), int(len_Up*train_test_scale), int(len_Down*train_test_scale), int(len_Left*train_test_scale), int(len_Right*train_test_scale), int(len_On*train_test_scale), int(len_Off*train_test_scale), int(len_Stop*train_test_scale), int(len_Go*train_test_scale)]
print (len_train1)

len_background_noise = 6
len_bed = 1713
len_bird = 1731
len_cat = 1733
len_dog = 1746
len_eight = 2352
len_five = 2357
len_four = 2372
len_happy = 1742
len_house = 1750
len_marvin = 1746
len_nine = 2364
len_one = 2370
len_seven = 2377
len_sheila = 1734
len_six = 2369
len_three = 2356
len_tree = 1733
len_two = 2373
len_wow = 1745
len_zero = 2376

len_train2 = [int(len_background_noise*train_test_scale), int(len_bed*train_test_scale), int(len_bird*train_test_scale), int(len_cat*train_test_scale), int(len_dog*train_test_scale), int(len_eight*train_test_scale), int(len_five*train_test_scale), int(len_four*train_test_scale), int(len_happy*train_test_scale), int(len_house*train_test_scale), int(len_marvin*train_test_scale), int(len_nine*train_test_scale), int(len_one*train_test_scale), int(len_seven*train_test_scale), int(len_sheila*train_test_scale), int(len_six*train_test_scale), int(len_three*train_test_scale), int(len_tree*train_test_scale), int(len_two*train_test_scale), int(len_wow*train_test_scale),int(len_zero*train_test_scale)]
print (len_train2)

sr = 16000  #44100  16000
Training_x = []
Training_y = []
Test_x = []
Test_y = []

Yes_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/yes/*/'))   #  label : 1
for i,orig_wav_file in enumerate(Yes_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[0] > 0:
        len_train1[0] = len_train1[0] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(1)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(1)
    
No_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/no/*/'))   #  label : 2
for i,orig_wav_file in enumerate(No_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[1] > 0:
        len_train1[1] = len_train1[1] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(2)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(2)    
    
Up_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/up/*/'))   #  label : 3
for i,orig_wav_file in enumerate(Up_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[2] > 0:
        len_train1[2] = len_train1[2] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(3)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(3)      
    
Down_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/down/*/'))   #  label : 4
for i,orig_wav_file in enumerate(Down_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[3] > 0:
        len_train1[3] = len_train1[3] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(4)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(4)      
    
    
Left_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/left/*/'))   #  label : 5
for i,orig_wav_file in enumerate(Left_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[4] > 0:
        len_train1[4] = len_train1[4] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(5)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(5)      
        
Right_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/right/*/'))   #  label : 6
for i,orig_wav_file in enumerate(Right_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[5] > 0:
        len_train1[5] = len_train1[5] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(6)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(6)     
        
On_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/on/*/'))   #  label : 7
for i,orig_wav_file in enumerate(On_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[6] > 0:
        len_train1[6] = len_train1[6] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(7)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(7)    
        
Off_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/off/*/'))   #  label : 8
for i,orig_wav_file in enumerate(Off_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[7] > 0:
        len_train1[7] = len_train1[7] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(8)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(8)          
        
Stop_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/stop/*/'))   #  label : 9
for i,orig_wav_file in enumerate(Stop_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[8] > 0:
        len_train1[8] = len_train1[8] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(9)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(9)          
        
Go_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/go/*/'))   #  label : 10
for i,orig_wav_file in enumerate(Go_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train1[9] > 0:
        len_train1[9] = len_train1[9] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(10)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(10)    
        
##############################################################################################################################################     
        
background_noise_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/_background_noise_/*/'))   #  label : 0
for i,orig_wav_file in enumerate(background_noise_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[0] > 0:
        len_train2[0] = len_train2[0] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)      
        
bed_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/bed/*/'))   #  label : 0
for i,orig_wav_file in enumerate(bed_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[1] > 0:
        len_train2[1] = len_train2[1] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)
        
bird_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/bird/*/'))   #  label : 0
for i,orig_wav_file in enumerate(bird_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[2] > 0:
        len_train2[2] = len_train2[2] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)
        
cat_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/cat/*/'))   #  label : 0
for i,orig_wav_file in enumerate(cat_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[3] > 0:
        len_train2[3] = len_train2[3] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)        

dog_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/dog/*/'))   #  label : 0
for i,orig_wav_file in enumerate(dog_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[4] > 0:
        len_train2[4] = len_train2[4] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)        

eight_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/eight/*/'))   #  label : 0
for i,orig_wav_file in enumerate(eight_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[5] > 0:
        len_train2[5] = len_train2[5] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)  
        
five_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/five/*/'))   #  label : 0
for i,orig_wav_file in enumerate(five_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[6] > 0:
        len_train2[6] = len_train2[6] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)  

four_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/four/*/'))   #  label : 0
for i,orig_wav_file in enumerate(four_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[7] > 0:
        len_train2[7] = len_train2[7] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)
        
happy_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/happy/*/'))   #  label : 0
for i,orig_wav_file in enumerate(happy_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[8] > 0:
        len_train2[8] = len_train2[8] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)

house_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/house/*/'))   #  label : 0
for i,orig_wav_file in enumerate(house_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[9] > 0:
        len_train2[9] = len_train2[9] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)

marvin_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/marvin/*/'))   #  label : 0
for i,orig_wav_file in enumerate(marvin_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[10] > 0:
        len_train2[10] = len_train2[10] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)

nine_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/nine/*/'))   #  label : 0
for i,orig_wav_file in enumerate(nine_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[11] > 0:
        len_train2[11] = len_train2[11] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)

one_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/one/*/'))   #  label : 0
for i,orig_wav_file in enumerate(one_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[12] > 0:
        len_train2[12] = len_train2[12] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)

seven_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/seven/*/'))   #  label : 0
for i,orig_wav_file in enumerate(seven_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[13] > 0:
        len_train2[13] = len_train2[13] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)
        
sheila_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/sheila/*/'))   #  label : 0
for i,orig_wav_file in enumerate(sheila_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[14] > 0:
        len_train2[14] = len_train2[14] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)
        
six_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/six/*/'))   #  label : 0
for i,orig_wav_file in enumerate(six_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[15] > 0:
        len_train2[15] = len_train2[15] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)        

three_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/three/*/'))   #  label : 0
for i,orig_wav_file in enumerate(three_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[16] > 0:
        len_train2[16] = len_train2[16] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)   

tree_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/tree/*/'))   #  label : 0
for i,orig_wav_file in enumerate(tree_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[17] > 0:
        len_train2[17] = len_train2[17] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)   

two_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/two/*/'))   #  label : 0
for i,orig_wav_file in enumerate(two_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[18] > 0:
        len_train2[18] = len_train2[18] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0)   

wow_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/wow/*/'))   #  label : 0
for i,orig_wav_file in enumerate(wow_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[19] > 0:
        len_train2[19] = len_train2[19] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0) 

zero_files = glob.glob(os.path.dirname('data/speech_commands_v0.01/zero/*/'))   #  label : 0
for i,orig_wav_file in enumerate(zero_files):
    truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
    if truncated_wav_vector.shape[0] > 16000:
        truncated_wav_vector = truncated_wav_vector[:16000]
    else:
        truncated_wav_vector = np.pad(truncated_wav_vector, (0, 16000-truncated_wav_vector.shape[0]), 'constant')
    
    if len_train2[20] > 0:
        len_train2[20] = len_train2[20] - 1
        Training_x.append(truncated_wav_vector)
        Training_y.append(0)
    else:
        Test_x.append(truncated_wav_vector)
        Test_y.append(0) 
        

print (len(Test_y))
print(np.shape(Test_x))
print (len(Training_y))
print(np.shape(Training_x))


datapath = os.path.join('data','COMMANDS_data.h5')
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_feature", dtype = 'float64', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)

datafile.create_dataset("Test_feature", dtype = 'float64', data=Test_x)
datafile.create_dataset("Test_label", dtype = 'int64', data=Test_y)
datafile.close()

print("Save data finish!!!")        