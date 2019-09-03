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
import main as mn
from keras.models import model_from_json
import os

def testRecord():
    print("Recording for one second")    
    fs = 44100  # Sample rate
    seconds = 2  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write("output.wav", fs, myrecording)
    # os.rename changes file location to the right folder however, it tends to corrupt the .WAV file
    #os.rename('/Users/hamzaehsan/Desktop/titleTownTech/speechRecognition/output.wav', '/Users/hamzaehsan/Desktop/titleTownTech/speechRecognition/Predict/Prediction/output.wav')
    print("output.wav has been saved to project directory")

def start():
        print("PRESS RETURN TO RECORD")
        key = input("PRESS RETURN TO RECORD :")
        if key == "":
                testRecord()

start()