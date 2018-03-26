# coding: utf-8

# In[1]:


import librosa as lb
import os
import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
from scipy import misc
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from multiprocessing.pool import Pool
from functools import partial
from sklearn.utils import shuffle
import pickle

SAMPLE = 100
Fs         = 12000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 20


"""
function load_audio_filenames() :
Finds all the music files in separate folders and returns a list of filenames and their IDS.
Input: music_dir 
Output: files, track_ids

"""
def load_audio_filenames(music_dir):
    
    folders = [os.path.join(music_dir,folder) for folder in list(os.walk(music_dir))[0][1]]
    print(folders[0])
    filenames = [[os.path.join(folder,f) for f in list(os.walk(folder))[0][2]] for folder in folders]
    filenames = [item for sublist in filenames for item in sublist]
    files = [item for item in filenames]
    track_ids = [str(int(filename.split('\\')[-1].split('.')[0])) for filename in filenames]
    assert len(files) == len(track_ids), "Lengths should match."
    print("Number of files: " + str(len(files)))
    print("Audio filenames loaded. Here is a sample:")
    print(files[0:5])
    print(track_ids[0:5])
    return files, track_ids


"""
This function stores the raw audio files within the ./data/audio/ folder 

"""
def save_raw_audio(files, track_ids, all_track_ids, all_labels, sample = SAMPLE,save =False):
    assert len(files) == len(track_ids)
    assert len(all_track_ids) == len(all_track_ids)
    print("Number of audio files: %d" %len(files))
    print("Number of datapoints in the csv file: %d" %len(all_track_ids))
    failed_count = 0
    if save:
        csv_dic = {idx : label for (idx,label) in zip(all_track_ids,all_labels)} 
        
        for i in tqdm(range(len(files))):
            try:
                if not os.path.exists("./data/audio/audio_" +str(track_ids[i])+'_' + csv_dic[int(track_ids[i])]  + ".npy"):
                   
                    signal, sr = lb.load(files[i], sr=Fs)
                    n_sample = signal.shape[0]
                    n_sample_fit = int(DURA*Fs)
                    if n_sample < n_sample_fit:
                        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
                    elif n_sample > n_sample_fit:
                        signal = signal[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]
                    signal = np.reshape(signal,(-1))
                    savefilename = "./data/audio/audio_" +str(track_ids[i])+'_' + csv_dic[int(track_ids[i])]  + ".npy"
                    #print("Saving to %s"%savefilename)
                    np.save(open(savefilename,"wb"),signal)
            except:
                failed_count += 1
                print("ID %d not found." %int(track_ids[i]))    



"""
function load_ids_labels():

Takes the file path of the csv file holding the small_tracks metadata and returns the
track_ids and their corresponding labels.
Input:
filepath :- str
output :- list(int), list(str)
"""
def load_ids_labels(csv_filepath):
    tracks = pd.read_csv(csv_filepath)
    ##Set new columns for the dataframe and remove the multi-index.
    new_cols = tracks.iloc[0]
    tracks = tracks.iloc[1:]
    new_cols[0] = "track_id"
    tracks.columns = new_cols
    #Track id column should be integer type, and the labels should be of string type.
    track_ids = tracks.track_id.astype(int).tolist()
    labels = tracks["genre_top"].astype(str).tolist()
    assert len(track_ids) == len(labels)
    print("%d audio ids and labels were loaded from the csv file." %len(track_ids))
    return track_ids,labels

def draw_sample(datafolder,sample):
    filenames = list(range(len(list(os.walk(datafolder))[0][2])))
    np.random.shuffle(filenames)
    filenames = filenames[0:sample]
    audio_vec = []
    labels = []
    track_ids = []
    for i,f in enumerate(filenames):
        signal, sr = lb.load(os.path.join(datafolder, f), sr=Fs)
        n_sample = signal.shape[0]
        n_sample_fit = int(DURA*Fs)
        if n_sample < n_sample_fit:
            signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
        elif n_sample > n_sample_fit:
            signal = signal[(n_sample-n_sample_fit)//2:(n_sample+n_sample_fit)//2]
        signal = np.reshape(signal,(-1))
        audio_vec.append(signal)
        track_ids.append(int(f.split('_')[1]))
        labels.append(f.split('_')[2].split('.')[0])
    assert len(audio_vec) == len(track_ids) == len(labels) == sample
    return audio_vec, track_ids, labels


"""
Saves the train and test partitions as IDs in a CSV file.
While loading within the batch function, this csv file is loaded
and then actual npy files will be loaded on the fly using the ID information.
This is to save memory, as this task is extremely memory intensive.

The datapoints are paired up and stored in the verification format.

"""
def save_verification_dataset(folder,savefilename,override = True,sample = 1000,num_test = 10000):
    #folder should be where your data is located directly.
    if override:

        filenames = [f for f in list(os.walk(folder))[0][2] if '.npy' in f]
        
        n = len(filenames)
        #Draw a stratified sample of the data with respect to the labels.
        #Maintains balance in data for all genre labels.
        ids = [f.split('_')[1] for f in filenames]
        labels = [f.split('_')[2] for f in filenames]
        _, sample_ids, _ , sample_labels = train_test_split(ids, labels, test_size = sample/n, 
            stratify = labels,random_state = 42)
        from collections import Counter
        print(Counter(sample_labels))
        del ids, labels ##Clear memory
        
        

        #Now get the filenames corresponding to the sample_ids and sample_labels
        filenames = [f for f in filenames if f.split('_')[1] in sample_ids] ##O(n^2) :/ Improve later.
        #Update n
        n = len(filenames)
        X1 = []
        X2 = []
        indicators = []
        labels1 = []
        labels2 = []
        count = 0
        for i in tqdm(range(0,n)):
            for j in range(i+1,n):
                
                X1.append(filenames[i].split('_')[1])
                X2.append(filenames[j].split('_')[1])
                
                labels1.append(filenames[i].split('_')[2].split('.')[0])
                labels2.append(filenames[j].split('_')[2].split('.')[0])
                if filenames[i].split('_')[2].split('.')[0] ==  filenames[j].split('_')[2].split('.')[0]:
                    indicators.append(0)
                else:
                    indicators.append(1)
                count += 1
        assert len(X1) == len(X2) == len(indicators)
        
        ##Save it as a csv file with 2 IDs, an indicator and the respective labels
        df = pd.DataFrame(data = {'X1_id': X1, 
            'X2_id': X2, 'Indicator':indicators, 
            'Label_1': labels1, 'Label_2': labels2})
        train, test = train_test_split(df, test_size=num_test/len(df), random_state = 42, stratify = df["Indicator"])
        #train, valid = train_test_split(train, test_size=num_test/len(train), random_state = 42, stratify = train["Indicator"])
        
        train.to_csv(os.path.join(folder, 'train_' + savefilename),index = False)
        #valid.to_csv(os.path.join(folder, 'valid_' + savefilename),index = False)
        test.to_csv(os.path.join(folder, 'test_' + savefilename),index = False)

        print("CSV files stored in %s as %s and %s."%(folder,'train_' + savefilename,'test_' + savefilename))
        return train, test

    else:
        train = pd.read_csv(os.path.join(folder, 'train_' + savefilename)) 
        #valid = pd.read_csv(os.path.join(folder, 'valid_' + savefilename)) 
        test =  pd.read_csv(os.path.join(folder, 'test_' + savefilename))
        return train, test


"""
This function is the most used outside this file.
Accepts a batch size, type of dataset, and looks through the CSV files with ID information
construct a generator that can repeatedly called to release batches of data.

SAMPLE parameter can be set in order to balance the data. Since there isn't a lot of 
similar-genre samples, this can be used to return batches from an equally balanced
shuffled dataset.
"""

def next_verif_batch(batch_size,type = 'mel',data = 'train', SAMPLE = 10000):
    if type == 'mel':
        current_dir = "./data/melfeatures/"
        if data == 'train':
            verif_file_path = "train_mel_verif.csv"
        else:
            verif_file_path = "test_mel_verif.csv"
        prefix = "melspect"
    else:
        current_dir = "./data/audio/"
        if data == "train":
            verif_file_path = "train_audio_verif.csv"
        else:
            verif_file_path = "test_audio_verif.csv"
        prefix = "audio"

    #Load the data and balance the classes
    verif_data_info = pd.read_csv(os.path.join(current_dir,verif_file_path))
    verif_0 = verif_data_info.loc[verif_data_info['Indicator'] == 0]
    verif_1 = verif_data_info.loc[verif_data_info['Indicator'] == 1]
    verif_same = verif_0.sample(SAMPLE)
    verif_diff = verif_1.sample(SAMPLE)
    
    verif_data_info =  pd.concat([verif_same, verif_diff], axis=0, join='inner')
    
    #Shuffle the dataframe
    verif_data_info = shuffle(verif_data_info)
    X1_ids = verif_data_info["X1_id"].tolist()
    X2_ids = verif_data_info["X2_id"].tolist()
    indicators = verif_data_info["Indicator"].astype(int).tolist()
    labels1 = verif_data_info["Label_1"].tolist()
    labels2 = verif_data_info["Label_2"].tolist()


    #Need the shape
    datafilepath = prefix + '_' + str(X1_ids[0]) + '_'  + labels1[0]+'.npy'
    datafile = os.path.join(current_dir,datafilepath)
    cur_features_file = load_from_file(datafile)
    n_mels = cur_features_file.shape[0]
    mel_vals = cur_features_file.shape[1]
    features_dic = {} #To store arrays and not repeat load them
    
    start = 0
    end = batch_size
    for k in range(len(X1_ids)//batch_size):
        
        start = k*batch_size
        end = min(start + batch_size, len(X1_ids))
        X1_cur_ids = X1_ids[start:end]
        X2_cur_ids = X2_ids[start:end]
        indicators_cur = indicators[start:end]
        X1_batch = np.zeros((len(X1_cur_ids),n_mels,mel_vals))
        X2_batch = np.zeros((len(X1_cur_ids),n_mels,mel_vals))
        indicator_batch = []
        cnt = 0
        for i in range(start,end):
            if not X1_ids[i] in features_dic.keys():
                datafilepath = prefix + '_' + str(X1_ids[i]) + '_'  + labels1[i]+'.npy'
                datafile = os.path.join(current_dir,datafilepath)
                cur_features_file = load_from_file(datafile)
                #Min max normalization
                cur_features_file = (cur_features_file - np.amin(cur_features_file))/(np.amax(cur_features_file) - np.amin(cur_features_file))
                features_dic[X1_ids[i]] = cur_features_file
            
            X1_batch[cnt,:,:] = features_dic[X1_ids[i]]

            if not X2_ids[i] in features_dic.keys():
                datafilepath = prefix + '_' + str(X2_ids[i]) + '_'  + labels2[i]+'.npy'
                datafile = os.path.join(current_dir,datafilepath)
                cur_features_file = load_from_file(datafile)
                #Min max normalization
                cur_features_file = (cur_features_file - np.amin(cur_features_file))/(np.amax(cur_features_file) - np.amin(cur_features_file))
                features_dic[X2_ids[i]] = cur_features_file
            X2_batch[cnt,:,:] = features_dic[X2_ids[i]]
            if indicators[i] == 0:
                indicator_batch.append([1,0])
            else:
                indicator_batch.append([0,1])
            cnt += 1
        X1_batch = np.reshape(X1_batch,(-1,n_mels,mel_vals,1)).astype('float32')
        X2_batch = np.reshape(X2_batch, (-1,n_mels,mel_vals,1)).astype('float32') 
        indicator_batch = np.asarray(indicator_batch).astype('float32')
        #Mean normalize
        X1_batch = (X1_batch - X1_batch.mean()) / X1_batch.std()
        X2_batch = (X2_batch - X2_batch.mean()) / X2_batch.std()
        yield (X1_batch,X2_batch,indicator_batch)


"""
Loads a npy file
"""
def load_from_file(filepath):
    return np.load(open(filepath,'rb'))

"""

Loads the entire dataset at once.
sample can take an int value for sample size
enter sample = 'all' for the full dataset
type = 'mel' to use melspectrogram feature extractor
type = 'audio' is not yet implemented

"""
def load_full_dataset(filepath, type = 'mel',sample = 100):
    if type == 'mel':
        current_dir = "./data/melfeatures/"
        prefix = "melspect"
    else:
        current_dir = "./data/audio/"
        prefix = "audio"
    verif_data_info = pd.read_csv(os.path.join(current_dir,filepath))
    #Shuffle the dataframe
    verif_data_info = shuffle(verif_data_info)
    X1_ids = verif_data_info["X1_id"].tolist()
    X2_ids = verif_data_info["X2_id"].tolist()
    indicators = verif_data_info["Indicator"].astype(int).tolist()
    labels1 = verif_data_info["Label_1"].tolist()
    labels2 = verif_data_info["Label_2"].tolist()
    num_points = len(X1_ids)
    if sample == 'all':
        sample = len(X1_ids)

    #Need the shape
    datafilepath = prefix + '_' + str(X1_ids[0]) + '_'  + labels1[0]+'.npy'
    datafile = os.path.join(current_dir,datafilepath)
    cur_features_file = load_from_file(datafile)

    features_dic = {} #For memoization of arrays
    if type == 'mel':
        prefix = 'melspect'
        n_mels = cur_features_file.shape[0]
        mel_vals = cur_features_file.shape[1]
        X1_batch = np.zeros((num_points,n_mels,mel_vals))
        X2_batch = np.zeros((num_points,n_mels,mel_vals))
        
    else:
        prefix = 'audio'
        audio_len = cur_features_file.shape[0]
        X1_batch = np.zeros((num_points,audio_len,1))
        X2_batch = np.zeros((num_points,audio_len, 1))
    indicator_batch = []

    for i in range(len(X1_ids)):
        
        if not X1_ids[i] in features_dic.keys():
            datafilepath = prefix + '_' + str(X1_ids[i]) + '_'  + labels1[i]+'.npy'
            datafile = os.path.join(current_dir,datafilepath)
            cur_features_file = load_from_file(datafile)

            #Min max normalization
            cur_features_file = (cur_features_file - np.amin(cur_features_file))/(np.amax(cur_features_file) - np.amin(cur_features_file))
            features_dic[X1_ids[i]] = cur_features_file
        
        if type == 'mel':
            X1_batch[i,:,:] = features_dic[X1_ids[i]]
        else:
            X1_batch[i,:,:] = np.reshape(features_dic[X1_ids[i]],(1,-1,1))
        if not X2_ids[i] in features_dic.keys():
            datafilepath = prefix + '_' + str(X2_ids[i]) + '_'  + labels2[i]+'.npy'
            datafile = os.path.join(current_dir,datafilepath)
            
            cur_features_file = load_from_file(datafile)
            #Min max normalization
            cur_features_file = (cur_features_file - np.amin(cur_features_file))/(np.amax(cur_features_file) - np.amin(cur_features_file))
            features_dic[X2_ids[i]] = cur_features_file
        if type == 'mel':
            X2_batch[i,:,:] = features_dic[X2_ids[i]]
        else:
            X2_batch[i,:,:] = np.reshape(features_dic[X2_ids[i]],(1,-1,1))
        if indicators[i] == 0:
            indicator_batch.append([1,0])
        else:
            indicator_batch.append([0,1])
    X1_batch = np.reshape(X1_batch[0:sample],(-1,n_mels,mel_vals,1)).astype('float32')
    X2_batch = np.reshape(X2_batch[0:sample], (-1,n_mels,mel_vals,1)).astype('float32') 
    indicator_batch = np.asarray(indicator_batch[0:sample]).astype('float32')

    return X1_batch, X2_batch, indicator_batch


"""
Finds the logscale melspectrogram features for a particular audio file
"""
def log_scale_melspectrogram(path, plot=False):
    
    signal = np.load(open(path, 'rb'))
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)
    
    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[round((n_sample-n_sample_fit)/2):round((n_sample+n_sample_fit)/2)]
    
    melspect = lb.amplitude_to_db(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        plt.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        print(melspect.shape)

    return melspect


"""
Extracts melspectrogram features and stores these in npy arrays with the 
names of these arrays being in the format melspect_<id>_<label>
"""
def save_mel_features(datafolder, load = False):
    filenames = list(os.walk(datafolder))[0][2]

    for i in tqdm(range(len(filenames))):
        savefilename = 'melspect_' + filenames[i].split('_')[1] + '_' + filenames[i].split('_')[2]
        if not os.path.exists(os.path.join(datafolder,'melfeatures',savefilename)):
            mel = log_scale_melspectrogram(os.path.join(datafolder, filenames[i]), plot = False)
            #Save this melfeature
            if not os.path.exists(os.path.join(datafolder,'melfeatures')):
                os.mkdir(os.path.join(datafolder,'melfeatures'))
            
            np.save(open(os.path.join(datafolder,'melfeatures',savefilename),'wb'), mel)
    print("Melspectrograms were saved in " + os.path.join(datafolder,'melfeatures'))



if __name__ == "__main__":
    melfolder = './data/melfeatures/'
    audiofolder = './data/audio/'
    all_track_ids, all_labels = load_ids_labels("./tracks_small.csv")  
    #audio_files , audio_track_ids = load_audio_filenames("./music_samples/")  
    #save_raw_audio(audio_files, audio_track_ids,all_track_ids, all_labels,  sample = 3, save = True)
    #save_mel_features(datafolder)
    #Get the verification IDs

    #Load the melspectrum labels
    #save_verification_dataset(melfolder, 'mel_verif.csv',override = True,sample = 1000)
    #Load the audio dataset
    #audio_verif_ids_labels = save_verification_dataset(audiofolder, 'audio_verif.csv',override = False,sample = 500)
    
    