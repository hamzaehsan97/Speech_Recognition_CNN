# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:25:32 2019

@author: HamzaEhsan
"""

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


def funct():
    wandb.init()
    config = wandb.config

    config.max_len = 11
    config.buckets = 20

    # Save data to array file first
    save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

    labels=["bed", "happy", "cat"]
    X_train, X_test, y_train, y_test = get_train_test()
    # # Feature dimension
    channels = 1
    config.epochs = 50
    config.batch_size = 100

    num_classes = 3
    X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
    X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)
    plt.imshow(X_train[100, :, :, 0])
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
    X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)


    x_final_vector = []
    x_final_vector_vector = []
    x_final_vector = wav2mfcc('output.wav')
    x_final_vector_vector.append(x_final_vector)
    x_final = np.array(x_final_vector_vector)
    # print predict sample shape
    print(x_final.shape)
    # x_final = x_final.reshape(config.buckets, config.max_len) 

    model = Sequential()
    input_shape = X_train[0].shape
    # print predict sample shape
    print(input_shape)

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])

    wandb.init()
    model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])


    y_final_oneHotEncoded= model.predict_classes(x_final, batch_size=1, verbose=0)
    y_final_num = argmax(y_final_oneHotEncoded)
    print(y_final_num)
    plot_model(model, to_file="model.png")