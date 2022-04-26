import os
import tensorflow_addons as tfa
import tensorflow as tf
from PIL import Image
from random import randint

from keras import layers
from tensorflow import keras

import model

root_path = "./data/images"

targetSize = 300
upscaleFactor = 2
inputSize = targetSize // upscaleFactor

testPath = root_path + "/test"

# Creates a list with the paths of all images in the testing subset and then sorts it

# Loading All Test Images
listOftestImgs = []

for img in os.listdir(testPath):
    listOftestImgs.append(os.path.join(testPath, img))

testImages = sorted(listOftestImgs)

# Loading Val and Training Images
# batch size 64 uses 29gb of ram
trainRawData = tf.keras.preprocessing.image_dataset_from_directory(root_path, label_mode=None,
                                                                   image_size=(targetSize, targetSize),
                                                                   validation_split=0.2, subset="training", seed=42, batch_size=64)
valRawData = tf.keras.preprocessing.image_dataset_from_directory(root_path, label_mode=None,
                                                                 image_size=(targetSize, targetSize),
                                                                 validation_split=0.2, subset="training", seed=42, batch_size=64)


def blur(img):
    return tfa.image.gaussian_filter2d(img)


# Random Flip
def randomVertFlip(img):
    return tf.image.random_flip_up_down(img)


# Random Flip
def randomHorFlip(img):
    return tf.image.random_flip_left_right(img)


# Add JPG Distort
def jpgDistort(img):
    return tf.image.random_jpeg_quality(img, 75, 100)


# Turn pixels values from 0 to 1
def normalizePixels(img):
    return img / 255.0


# Convert To YUV Colourspace Format
def toYUV(img):
    img = tf.image.rgb_to_yuv(img)  # Change the image format to yuv scale,
    return tf.split(img, 3, axis=3)[0]


# Apply Transformations Individually Because TensorFlow Doesn't Support Batches For These
# https://stackoverflow.com/questions/38920240/tensorflow-image-operations-for-batches


# Randomly Flip Data First
trainRandFlip = trainRawData.map(randomVertFlip)
valRandFlip = valRawData.map(randomVertFlip)

# Then Distort Flip Before Converting Colour Space
#trainJPGDist = trainRandFlip.map(lambda img: jpgDistort(img))
# valJPGDist = valRandFlip.map(lambda img: jpgDistort(img))

# Normalizing pixel values from 0 to 1 for smaller complexity for val and train
trainScaledData = trainRandFlip.map(normalizePixels)
valScaledData = valRandFlip.map(normalizePixels)


def scaleToSize(img, targetSize):
    return tf.image.resize(img, [targetSize, targetSize], method="area")


#
# # Convert Image to YUV Format
trainDataYUV = trainScaledData.map(toYUV)
valDataYUV = valScaledData.map(toYUV)

# Scale Down Image To Input Size | target Size // Scale Factor
trainDataScaled = trainDataYUV.map(lambda img: (scaleToSize(img, inputSize), img))
valDataScaled = valDataYUV.map(lambda img: (scaleToSize(img, inputSize), img))

trainData = trainDataScaled.prefetch(buffer_size=32)
valData = valDataScaled.prefetch(buffer_size=32)

# l = list(trainData)
#
# for i in range(5):
#     img1 = tf.keras.preprocessing.image.array_to_img(l[i][1][0])
#
#     img1.show()
#
#     img2 = tf.keras.preprocessing.image.array_to_img(l[i][0][0])
#     img2.show()

loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
callbacks = [early_stopping_callback]

modelMaker = model.model1(upscaleFactor)
modelMaker.compile(optimizer=optimizer, loss=loss_function)

modelMaker.fit(trainData, epochs=10, validation_data=valData, verbose=1, callbacks=callbacks, batch_size=64)

modelMaker.save("weights.h5")
