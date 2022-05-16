import numpy
import tensorflow as tf
import os
import numpy as np
import cv2
import random

# Generate Training Data
from PIL import Image
from keras_preprocessing import image

# Image Augmentation
import skimage.filters
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import variables

# Keep This Off If Not Using File
# Allows For Quick Numpy Interaction

if variables.usingDataGen:
    tf.enable_eager_execution()
    print("Eager Execution Enable -Using DataGen")

trainingSize = variables.targetSize
scaler = variables.scaleFactor

saveAugmentedTrainingImages = True
augmentSavePath = variables.augmentSavePath

loadOriginal = variables.loadOriginal

highRes = []
lowRes = []


# Load Data
def loadData(direc, output):
    count = 0

    for i in os.listdir(direc):
        count += 1

        img = image.load_img(direc + fr"\{i}", target_size=(trainingSize, trainingSize, 3))
        img = image.img_to_array(img)

        output.append(img)

        print(f"Loaded: {count} / {len(os.listdir(direc))} | {i}")

    print("Finished Loading Data")
    return output


# Image Distortions and Augmentation
def jpgDist(img, minVal, maxVal):
    return tf.image.random_jpeg_quality(img, minVal, maxVal)
    # return tf.image.adjust_jpeg_quality(img, random.randint(99, 100))


# Higher Sigma = More Blur
# https://stackoverflow.com/a/59286930/11521629
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, sigma):
    blur = _gaussian_kernel(3, sigma, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def randomBlur(img, sigma):
    return apply_blur(img, sigma)


def downSample(img, lowResDim, scalingMethod):
    return tf.image.resize(img, (lowResDim, lowResDim), method=scalingMethod)


def upSample(img, fullResDim, scalingMethod):
    return tf.image.resize(img, (fullResDim, fullResDim), method=scalingMethod)


# https://stackoverflow.com/a/30609854/11521629
def randomNoise(img, noise_typ):
    shape = img.shape

    # Adding Gaussian noise
    noise = tf.random_normal(shape=shape, mean=0.0, stddev=noise_typ, dtype=tf.float32)
    return img + noise


# Append All Image Paths To List
# First Append Full Res 300x300 Data
loadData(loadOriginal, highRes)

# Do To Training Data
# Effects Are Random
#
# Apply Effects
# Scale Down
# Apply Effects
# Scale Up Using Nearest Neighbour

# Then To Both
# normalize pixels
# convert to yuv

# Augment Training Data
for i in range(len(highRes)):
    print(f"Augmenting {i} / {len(highRes)}")

    img = highRes[i]

    # Using Real-Esrgan Model Approach To Data Augmentation
    # 2nd order degradation, data is degraded 2 times
    # JPG Distortion is broken and doesn't work for some reason

    # for j in range(0, 2):
    #     # 33% Chance To Blur
    #
    #     if random.randint(0, 2) == 0:
    #         img = randomBlur(img, random.randint(1, 6))
    #
    #     # Down sample Using Bicubic For Best Results
    #     # Then Upscale
    #     if j == 0:
    #         img = downSample(img, trainingSize // scaler, ResizeMethod.BICUBIC)
    #
    #     if j == 1:
    #         img = upSample(img, trainingSize, ResizeMethod.BICUBIC)
    #
    #     if random.randint(0, 2) == 0:
    #         img = randomNoise(img, random.uniform(0, 0.3))

    # Scale Back Down To Prevent It Doing It in The Other Loop
    img = downSample(img, trainingSize // scaler, ResizeMethod.BICUBIC)

    # Save Degraded Image For Training

    # print(type(img))
    lowRes.append(img)

    # Save Augmented Images
    if saveAugmentedTrainingImages:
        # print(img)
        # print(img.shape)

        # Cannot make numpy array numpy array so have to filter for tensors only
        # If not a numpy array then convert it to one and save
        if not isinstance(img, numpy.ndarray):
            array = img.numpy()
            Image.fromarray(array.astype(np.uint8)).save(augmentSavePath + fr"\{os.listdir(loadOriginal)[i]}")

        else:
            Image.fromarray(img.astype(np.uint8)).save(augmentSavePath + fr"\{os.listdir(loadOriginal)[i]}")

        print(f"Saved: {os.listdir(loadOriginal)[i]}")

# Normalize and Convert To YUV Format
# Done In Main File So Data Can Be Adjusted Accordingly
