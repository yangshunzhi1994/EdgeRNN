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

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

len_happy = 0
len_angry = 0
len_sad = 0
len_neutral = 0
train_test_scale = 0.8


for sess in range(1, 6):
    emo_evaluation_dir = 'data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
    evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
    for file in evaluation_files:
        with open(emo_evaluation_dir + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            
            if emotion == 'hap' or emotion == 'exc':
                len_happy = len_happy + 1
                emotion = 'hap'
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
            
            if emotion == 'ang':
                len_angry = len_angry + 1
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
                
            if emotion == 'sad':
                len_sad = len_sad + 1
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
                
            if emotion == 'neu':
                len_neutral = len_neutral + 1
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
                
            
print("len_happy：%d, len_angry：%d,len_sad：%d, len_neutral: %d" % (len_happy, len_angry, len_sad, len_neutral))

len_train = [int(len_happy*train_test_scale), int(len_angry*train_test_scale), int(len_sad*train_test_scale), int(len_neutral*train_test_scale)]

print (len_train)

df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions
df_iemocap['val'] = vals
df_iemocap['act'] = acts
df_iemocap['dom'] = doms

df_iemocap.tail()
df_iemocap.to_csv('data/IEMOCAP_full_release/df_iemocap.csv', index=False)

labels_df = pd.read_csv('data/IEMOCAP_full_release/df_iemocap.csv')
iemocap_dir = 'data/IEMOCAP_full_release/'


sr = 16000  #44100  16000
Training_x = []
Training_y = []
Test_x = []
Test_y = []

emotion_dict = {'hap': 0,
                'ang': 1,
                'sad': 2,
                'neu': 3}

for sess in range(1, 6):  # using one session due to memory constraint, can replace [5] with range(1, 6)
    orig_wav_files = glob.glob(os.path.dirname('{}Session{}/sentences/wav/*/*/'.format(iemocap_dir, sess)))
    for i,orig_wav_file in enumerate(orig_wav_files):
        try:
            truncated_wav_vector, _sr = librosa.load(orig_wav_file, sr=sr)
            short_wav_file, file_format = orig_wav_file.split("/",6)[6].split('.')
            for index, row in labels_df[labels_df['wav_file'].str.match(short_wav_file)].iterrows():
                emotion = row['emotion']
                
                if truncated_wav_vector.shape[0] > 92585:
                    truncated_wav_vector = truncated_wav_vector[:92585]
                else:
                    truncated_wav_vector = np.pad(truncated_wav_vector, (0, 92585-truncated_wav_vector.shape[0]), 'constant')
                
                if emotion == 'hap':
                    label = emotion_dict[emotion]
                    feature = truncated_wav_vector
                    if len_train[0] > 0:
                        len_train[0] = len_train[0] - 1
                        Training_x.append(feature)
                        Training_y.append(int(label))
                    else:
                        Test_x.append(feature)
                        Test_y.append(int(label))
                            
                if emotion == 'ang':
                    label = emotion_dict[emotion]
                    feature = truncated_wav_vector
                    if len_train[1] > 0:
                        len_train[1] = len_train[1] - 1
                        Training_x.append(feature)
                        Training_y.append(int(label))
                    else:
                        Test_x.append(feature)
                        Test_y.append(int(label))
                            
                if emotion == 'sad':
                    label = emotion_dict[emotion]
                    feature = truncated_wav_vector
                    if len_train[2] > 0:
                        len_train[2] = len_train[2] - 1
                        Training_x.append(feature)
                        Training_y.append(int(label))
                    else:
                        Test_x.append(feature)
                        Test_y.append(int(label))
                            
                if emotion == 'neu':
                    label = emotion_dict[emotion]
                    feature = truncated_wav_vector
                    if len_train[3] > 0:
                        len_train[3] = len_train[3] - 1
                        Training_x.append(feature)
                        Training_y.append(int(label))
                    else:
                        Test_x.append(feature)
                        Test_y.append(int(label))
                
                                
        except:
            print('An exception occured for {}'.format(orig_wav_file))
                       
print (len(Test_y))
print(np.shape(Test_x))
print (len(Training_y))
print(np.shape(Training_x))


datapath = os.path.join('data','IEMOCAP_data.h5')
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_feature", dtype = 'float64', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)

datafile.create_dataset("Test_feature", dtype = 'float64', data=Test_x)
datafile.create_dataset("Test_label", dtype = 'int64', data=Test_y)
datafile.close()

print("Save data finish!!!")