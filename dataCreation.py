import h5py
import numpy
import tensorflow as tf
import os
import numpy as np
import cv2
import random

# Generate Training Data
from PIL import Image
from keras_preprocessing import image
import threading

# Image Augmentation
import skimage.filters
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import augmentations
import config

# Allow For Quick Numpy Array Work
tf.enable_eager_execution()

# Check If Custom DataDir Is Specified
if len(config.dataDir) > 0:
    dataDir = config.dataDir
else:
    dataDir = "./data"

if len(config.saveAugImagePath) > 0:
    saveAugImagePath = config.saveAugImagePath
else:
    saveAugImagePath = f"{dataDir}"

# Check If Folders Exist

# Check If Images Exist

# Process Training And Validation Files
filesToDo = [dataDir + "/training/original", dataDir + "/validation/original"]
names = ["lowResTrain", "highResTrain", "lowResVal", "highResVal"]


# Do Training And Validation Data
def createData(j, doAug, lowResName, highResName):
    imgHighResYAr = []
    imgHighResUAr = []
    imgHighResVAr = []

    imgLowResYAr = []
    imgLowResUAr = []
    imgLowResVAr = []

    print("------------------------------")
    print("Currently Working On Dir: " + j)

    for i in os.listdir(j):
        print("Processing: " + i)

        # Load Image
        imgOriginal = load_img(f"{j}/{i}")
        imgOriginal = img_to_array(imgOriginal)
        imgLowRes = imgOriginal

        lowResX = imgLowRes.shape[0]
        lowResY = imgLowRes.shape[1]

        if doAug:

            # Do Augmentations On Full Size Image

            for k in range(0, 2):
                # 33% Chance To Blur

                if random.randint(0, 2) == 0:
                    imgLowRes = augmentations.randomBlur(imgLowRes, random.randint(1, 6))

                # Down sample Using Bicubic For Best Results
                # Then Upscale
                if k == 0:
                    imgLowRes = augmentations.downSample(imgLowRes, lowResX // config.upscalingFactor,
                                                         lowResY // config.upscalingFactor,
                                                         ResizeMethod.BICUBIC)

                if k == 1:
                    imgLowRes = augmentations.upSample(imgLowRes, lowResX, lowResY,
                                                       ResizeMethod.BICUBIC)

                if random.randint(0, 2) == 0:
                    imgLowRes = augmentations.randomNoise(imgLowRes, random.uniform(0, 0.3))

        # Scale To Training Size // Factor So Ai Can Scale Image
        imgLowRes = augmentations.downSample(imgLowRes, config.trainingSizeX // config.upscalingFactor,
                                             config.trainingSizeY // config.upscalingFactor, ResizeMethod.BICUBIC)

        # Make Sure Everything Is Numpy Array
        # try:
        imgLowRes = imgLowRes.numpy()
        # except:
        #     imgLowRes = Image.fromarray(imgLowRes.astype(np.uint8))

        # Save Augmented Images
        if config.saveAugImage:
            imgLowResIMAGE = Image.fromarray(imgLowRes.astype(np.uint8))
            imgLowResIMAGE.save(j.replace("/original", f"/augmented/{i}"))
            print(f"Saved Image Augmentation: {i}")

        imgHighRes = cv2.resize(imgOriginal, (config.trainingSizeX, config.trainingSizeY), interpolation=cv2.INTER_AREA)

        imgHighRes = cv2.cvtColor(imgHighRes, cv2.COLOR_RGB2YCrCb)
        imgLowRes = cv2.cvtColor(imgLowRes, cv2.COLOR_RGB2YCrCb)

        imgHighRes = imgHighRes / 255.
        imgLowRes = imgLowRes / 255.

        imgHighResY = imgHighRes[:, :, 0]
        imgHighResU = imgHighRes[:, :, 1]
        imgHighResV = imgHighRes[:, :, 2]

        imgLowResY = imgLowRes[:, :, 0]
        imgLowResU = imgLowRes[:, :, 1]
        imgLowResV = imgLowRes[:, :, 2]

        # Append Images To Array So They Can Be Saved To H5 File
        imgHighResYAr.append(imgHighResY)
        imgHighResUAr.append(imgHighResU)
        imgHighResVAr.append(imgHighResV)

        imgLowResYAr.append(imgLowResY)
        imgLowResUAr.append(imgLowResU)
        imgLowResVAr.append(imgLowResV)

    print("Saving Data As H5 File")

    dataLowY = np.array(imgLowResYAr, dtype=np.float16)
    dataLowU = np.array(imgLowResUAr, dtype=np.float16)
    dataLowV = np.array(imgLowResVAr, dtype=np.float16)

    dataHighY = np.array(imgHighResYAr, dtype=np.float16)
    dataHighU = np.array(imgHighResUAr, dtype=np.float16)
    dataHighV = np.array(imgHighResVAr, dtype=np.float16)

    with h5py.File(f'./imageDatah5/{lowResName}_dataLowY.h5', 'w') as h:
        h.create_dataset('data', data=dataLowY, shape=dataLowY.shape)

    with h5py.File(f'./imageDatah5/{lowResName}_dataLowU.h5', 'w') as h:
        h.create_dataset('data', data=dataLowU, shape=dataLowU.shape)

    with h5py.File(f'./imageDatah5/{lowResName}_dataLowV.h5', 'w') as h:
        h.create_dataset('data', data=dataLowV, shape=dataLowV.shape)

    with h5py.File(f'./imageDatah5/{highResName}_dataHighY.h5', 'w') as h:
        h.create_dataset('data', data=dataHighY, shape=dataHighY.shape)

    with h5py.File(f'./imageDatah5/{highResName}_dataHighU.h5', 'w') as h:
        h.create_dataset('data', data=dataHighU, shape=dataHighU.shape)

    with h5py.File(f'./imageDatah5/{highResName}_dataHighV.h5', 'w') as h:
        h.create_dataset('data', data=dataHighV, shape=dataHighV.shape)


# Start Threads To Do Val And Train Data At Same Time
x = threading.Thread(target=createData, args=(filesToDo[0], config.doAugmentations, names[0], names[1],))
x.start()

x = threading.Thread(target=createData, args=(filesToDo[1], config.doAugmentations, names[2], names[3],))
x.start()
