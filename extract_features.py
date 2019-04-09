"""
Created on Tue May  8 09:01:09 2018

@author: jsager

Computes a feature vector for a given audio file

"""

import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy

#main
def extract(filepath):
    #load file, get sample rate
    x, sr = librosa.load(filepath, sr=None)
    x = librosa.resample(x, sr, 22050)
    sample_rate = 22050

    #use a 50 ms frame for pitch (http://cs229.stanford.edu/proj2007/ShahHewlett%20-%20Emotion%20Detection%20from%20Speech.pdf)
    #use a 25 ms frame for mfccs
    pitch_frame_length = .05
    mfcc_frame_length = .025
    others_frame_length = .1
    
    pitches = _get_pitch(x, sample_rate, pitch_frame_length)    #1D of length #frames
    pitchstats = _get_pitchstats(pitches)                       #1D of length 10
    mfccs = _get_mfcc(x, sample_rate, mfcc_frame_length)        #13*#frames
    mfccstats = _get_mfcc_stats(mfccs)                          #length 104
    zcstats = _get_zerocross(x, sample_rate, others_frame_length)
    rmsestats = _get_rmse(x, sample_rate, others_frame_length)

    feature_vector = pitchstats + mfccstats + zcstats + rmsestats
    #feature_vector = mfccstats + zcstats + rmsestats
    
    #feature_vector = [round(elem, 2) for elem in feature_vector]
    
    #add male/female tag
#    if '11' in filepath or '22' in filepath or '33' in filepath or '44' in filepath or '55' in filepath or '66' in filepath:
#        feature_vector = feature_vector + [1.0]
#    else:
#        feature_vector = feature_vector + [0.0]

    return feature_vector

def _get_pitch(x, sr, pitch_frame_length):
    #split x into non overlaping frames of 50 ms
    samples_per_frame = math.floor(pitch_frame_length * sr)
    #frames = librosa.util.frame(y=x, frame_length=samples_per_frame, hop_length=samples_per_frame)
    pitch_vector = []

    #compute piptrack for each of the frames, and add the max to the vector of pitches
    pitches, magnitudes = librosa.core.piptrack(y=x, sr=sr, fmin=85, fmax=1600, hop_length=samples_per_frame)
    for i in range(0, pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_vector.append(pitches[index, i])

    pitch_vector = list(map(float, pitch_vector))
    rounded = [round(elem, 2) for elem in pitch_vector]

    return rounded

def _get_pitchstats(pitches):
    #Mean, Median, Variance, Maximum, Minimum (for the pitch vector and its derivative)
    #average energies of voiced and unvoiced
    #speaking rate (inverse of average length of voiced part)
    #http://cs229.stanford.edu/proj2007/ShahHewlett%20-%20Emotion%20Detection%20from%20Speech.pdf
    
    #7/8 - added kurtosis, added skewness, added range, used librosa's delta, removed median
    #dpitches = np.gradient(pitches)
    dpitches = librosa.feature.delta(pitches)
    
    statvector = [statistics.mean(pitches), statistics.variance(pitches), max(pitches),  
        statistics.mean(dpitches), statistics.variance(dpitches), max(dpitches), min(dpitches), scipy.stats.kurtosis(pitches), scipy.stats.skew(pitches),scipy.stats.kurtosis(dpitches), scipy.stats.skew(dpitches), max(pitches) - min(pitches), max(dpitches) - min(dpitches)]

    return statvector

def _get_mfcc(x, sr, frame_length):
    #use 25 ms frames
    samples_per_frame = math.floor(frame_length * sr)

    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13, hop_length=samples_per_frame)



    for i in range(0, len(mfccs[0])):
        mfccs[:,i] = list(map(float, mfccs[:,i]))

    #rounded = [round(elem, 2) for elem in mfccs]
#    plt.plot(mfccs[12,:])
#    plt.show()
    return mfccs

def _get_mfcc_stats(mfccs):
    #13 * no. frames
    statvector = []

    #compute stats for each of 13 mfcc and each mfcc's derivative
    for i in range(0, len(mfccs)):
        column = list(map(float, mfccs[i]))
        dcolumn = librosa.feature.delta(column)
        
        statvector.extend([statistics.mean(column), statistics.variance(column), max(column), min(column), 
            statistics.mean(dcolumn), statistics.variance(dcolumn), max(dcolumn), min(dcolumn)])
        
    return statvector
    
def _get_zerocross(x, sr, frame_length):
    samples_per_frame = math.floor(frame_length * sr)
    
    zero_crossings = librosa.feature.zero_crossing_rate(y=x, frame_length = samples_per_frame, hop_length = math.floor(samples_per_frame/2))

    zero_crossings = list(map(float, zero_crossings[0]))
    
    dzero_crossings = librosa.feature.delta(zero_crossings)
    
    statvector = [statistics.mean(zero_crossings), statistics.variance(zero_crossings), max(zero_crossings),  
        statistics.mean(dzero_crossings), statistics.variance(dzero_crossings), max(zero_crossings), min(dzero_crossings), scipy.stats.kurtosis(zero_crossings), scipy.stats.skew(zero_crossings),scipy.stats.kurtosis(dzero_crossings), scipy.stats.skew(dzero_crossings), max(zero_crossings) - min(zero_crossings), max(dzero_crossings) - min(dzero_crossings)]
    return statvector

    
def _get_rmse(x, sr, frame_length):
    samples_per_frame = math.floor(frame_length * sr)

    energy = librosa.feature.rmse(y=x, frame_length = samples_per_frame, hop_length = math.floor(samples_per_frame/2))
    
    energy = list(map(float, energy[0]))
    
    denergy = librosa.feature.delta(energy)

    statvector = [statistics.mean(energy), statistics.variance(energy), max(energy),  
        statistics.mean(denergy), statistics.variance(denergy), max(denergy), min(denergy), scipy.stats.kurtosis(energy), scipy.stats.skew(energy),scipy.stats.kurtosis(denergy), scipy.stats.skew(denergy), max(energy) - min(energy), max(denergy) - min(denergy)]
    return statvector











#print(len(extract('/home/jsager/Desktop/NEW/raw audio/test/angry/x6-5')))
