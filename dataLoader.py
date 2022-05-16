import h5py
import os
import cv2
import numpy as np

import variables

targetSize = variables.targetSize
scaler = variables.scaleFactor


def loadData(path, targetSize, scaler, lowRes, outputName="defaultName", saveFile=False):
    files = os.listdir(path)

    counter = 1
    totalFiles = len(files)

    data = []

    for file in files:
        fpath = path + "/" + file

        img = cv2.imread(fpath, cv2.IMREAD_COLOR)

        print("Trying: " + fpath)

        # Image Already Scaled Before
        if not lowRes:
            # img = cv2.resize(img, (targetSize // scaler, targetSize // scaler), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (targetSize, targetSize), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        img = img / 255.

        img = img[:, :, 0]

        data.append(img)

        print(f"Done: {counter} / {totalFiles}")
        counter += 1

    data = np.array(data, dtype=np.float16)

    if saveFile:
        with h5py.File(f'{outputName}.h5', 'w') as h:
            h.create_dataset('data', data=data, shape=data.shape)


loadData("./data/train/lowRes", lowRes=True, targetSize=targetSize, scaler=scaler,
         outputName="trainLowRes", saveFile=True)

loadData("./data/val/lowRes", lowRes=True, targetSize=targetSize, scaler=scaler,
         outputName="valLowRes", saveFile=True)

loadData("./data/train/original", lowRes=False, targetSize=targetSize, scaler=scaler,
         outputName="trainHighRes", saveFile=True)

loadData("./data/val/original", lowRes=False, targetSize=targetSize, scaler=scaler,
         outputName="valHighRes", saveFile=True)
