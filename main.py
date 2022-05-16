import h5py
import tensorflow as tf
import os
from PIL import Image
from random import randint

import numpy as np
from tensorflow import keras
from keras_preprocessing import image

import upscalingModel
import variables

targetSize = variables.targetSize
upscaleFactor = variables.scaleFactor

trainHighResPath = "./trainHighRes.h5"
trainLowResPath = "./trainLowRes.h5"

valHighResPath = "./valHighRes.h5"
valLowResPath = "./valLowRes.h5"


def readFile(file, targetSize):

    f = h5py.File(file, 'r')
    data = f['data'][...]
    f.close()

    data = data.reshape((len(data)), targetSize, targetSize, 1)

    print(data.shape)

    return data


trainX = readFile(trainLowResPath, targetSize//upscaleFactor)
trainY = readFile(trainHighResPath, targetSize)

valX = readFile(valLowResPath, targetSize//upscaleFactor)
valY = readFile(valHighResPath, targetSize)

print("Starting Model")

loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
callbacks = [early_stopping_callback]


model = upscalingModel.model1(upscaleFactor)
# modelMaker.summary()


model.compile(optimizer=optimizer, loss=loss_function)

model.fit(trainX, trainY, epochs=variables.epochs, validation_data=(valX, valY),
          verbose=1, batch_size=32)

model.save(f"{variables.modelSaveName}.h5")
