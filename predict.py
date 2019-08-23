from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
from numpy import argmax
import matplotlib.pyplot as plt
from keras.utils import plot_model
import graphviz
from numpy import loadtxt
from keras.models import load_model

# wandb.init()
# config = wandb.config

# config.epochs = 50
# config.batch_size = 100


x_final_vector = []
x_final_vector_vector = []
x_final_vector = wav2mfcc('output.wav')
x_final_vector_vector.append(x_final_vector)
# x_final_vector_vector[0] = x_final_vector
x_final = np.array(x_final_vector_vector)
x_final = x_final.reshape(x_final.shape[0], 20, 11)
# print predict sample shape
print(x_final.shape)
# x_final = x_final.reshape(config.buckets, config.max_len) 

model=load_model("model.h5")

def predictRecording():
    y_final_oneHotEncoded= model.predict_classes(x_final, batch_size=1, verbose=0)
    y_final_prob= model.predict_proba(x_final, batch_size=1, verbose=0)
    y_final_prob= np.round(y_final_prob,2)
    predictNumber = y_final_oneHotEncoded[0]
    listNames = ["happy","bed","cat"]
    print(f"PREDICTION NUMBER = {predictNumber}")
    print(f"PREDICTION = {listNames[predictNumber]}")
    print(f"PREDICTION Prob= {y_final_prob}")

