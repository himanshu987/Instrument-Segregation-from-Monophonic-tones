import numpy as np
import sklearn
import librosa
import glob

def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    # this function first normalizes audio data
    #calculates the amplitude of each frame
    #silence_threshold is used to flip the silence part
    #the number of silence frame is returned.
    #trim_ms is the counter
    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms

def feature_extract():

    sr = 44100
    window_size = 2048
    hop_size = window_size/2
    data = []

    #read file
    files = glob.glob('tp/*.mp3')
    np.random.shuffle(files)
    for filename in files:

        music, sr= librosa.load(filename, sr = sr)

        start_trim = detect_leading_silence(music)
        end_trim = detect_leading_silence(np.flipud(music))

        duration = len(music)
        trimmed_sound = music[start_trim:duration-end_trim]
        # the sound without silence

        #use mfcc to calculate the audio features
        mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
        aver = np.mean(mfccs, axis = 1)
        feature = aver.reshape(20)

        #store label and feature
        #the output should be a list
        #label and feature, corresponds one by one
        #feature.append(aver)
        label=0
        """if filename[16:19] == 'cel':
            label = 1
        elif filename[16:19] == 'cla':
            label = 2"""
        if filename[3:6] == 'flu':
            label = 3
        elif filename[3:6] == 'vio':
            label = 4
        elif filename[3:6] =='tru':
            label = 5
        print('in pre file', filename[3:6])
        print('in pre label ', label)

        data2 = [filename, feature, label]
        # print data2
        # print feature.shape
        data.append(data2)
        #data = np.vstack((data, data2))
        # print data
    return data

def main():
    data = feature_extract()
    print data
    print len(data)