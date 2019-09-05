import sounddevice as sd
from scipy.io.wavfile import write
from predictProcess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
from numpy import argmax
import matplotlib.pyplot as plt
from keras.utils import plot_model
from numpy import loadtxt
from keras.models import load_model
from keras.models import model_from_json
import os
import wave
import contextlib


# fname = './data/bed/0a7c2a8d_nohash_0.wav'
# with contextlib.closing(wave.open(fname,'r')) as f:
#     frames = f.getnframes()
#     rate = f.getframerate()
#     duration = frames / float(rate)
#     print(duration)
#     print(frames)
#     print(rate)
num = 0

def testRecord():
    print("Recording for one second")    
    fs = 16000  # Sample rate
    seconds = 1  # Duration of recording
    #Change Range in the line below to record multiple recordings
    for num in range(0,1):
        print("PRESS RETURN TO RECORD")
        key = input("PRESS RETURN TO RECORD :")
        if key == "":
                myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()  # Wait until recording is finished
                name = "output{}.wav".format(num)
                write(name, fs, myrecording)
                # os.rename changes file location to the right folder however, it tends to corrupt the .WAV file
                # os.rename('/Users/hamzaehsan/Desktop/titleTownTech/speechRecognition/output.wav', '/Users/hamzaehsan/Desktop/titleTownTech/speechRecognition/Predict/Prediction/output.wav')
                print("output.wav has been saved to project directory")

testRecord()