import tensorflow as tf
import os
from PIL import Image
from random import randint

from tensorflow import keras
from tensorflow.keras.preprocessing import image

import upscalingModel

targetSize = 300
upscaleFactor = 2

trainHighResPath = r"D:\upscaler_v2\data\train"
trainLowResPath = r"D:\upscaler_v2\data\train"

valHighResPath = r"D:\upscaler_v2\data\val\original"
valLowResPath = r"D:\upscaler_v2\data\val\lowRes"


# Convert To YUV Colourspace Format
def toYUV(img):
    img = tf.image.rgb_to_yuv(img)  # Change the image format to yuv scale,
    return tf.split(img, 3, axis=3)[0]


def read(path):
    img = image.load_img(path, target_size=(targetSize, targetSize, 3))
    img = image.img_to_array(img)
    img = img / 255.
    img = img.map(toYUV)

    return img


trainHighResRawData = read(trainHighResPath)
trainLowResRawData = read(trainLowResPath)

valHighResRawData = read(valHighResPath)
valLowResRawData = read(valLowResPath)

# Defining Model Once Data Loaded In

loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
callbacks = [early_stopping_callback]

model = upscalingModel.model1(upscaleFactor)
# modelMaker.summary()


model.compile(optimizer=optimizer, loss=loss_function)

model.fit(trainLowResRawData, trainHighResRawData, epochs=10, validation_data=(valLowResRawData, valHighResRawData),
          verbose=1)

# weights one was 6 epoch

# freezes at 39 epoch for some reason
model.save("weightsMinecraftx2_Iter1.h5")
