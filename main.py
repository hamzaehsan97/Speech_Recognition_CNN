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



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])


save_data_to_array(max_len=11, n_mfcc=20)
x = np.load("prediction.npy")
x = x.reshape(x.shape[0], 20, 11, 1)
y_final_oneHotEncoded= model.predict_classes(x, batch_size=1, verbose=0)
print(y_final_oneHotEncoded)
y_final_prob= model.predict_proba(x, batch_size=len(x), verbose=0)
predictNumber = y_final_oneHotEncoded[0]
listNames = ["cat","bed","happy"]
print(f"PREDICTION NUMBER = {predictNumber}")
print(f"PREDICTION = {listNames[predictNumber]}")
print(f"PREDICTION Prob= {y_final_prob}")

