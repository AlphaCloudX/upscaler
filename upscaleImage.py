import cv2
import PIL
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import time
import os
import tensorflow as tf
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import upscalingModel
import config

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

targetSizeX = config.trainingSizeY
targetSizeY = config.trainingSizeY
upscaleFactor = config.upscalingFactor

# Most use x2 or x4 for 300

modelY = upscalingModel.model1(upscaleFactor)
modelY.load_weights(f"./modelWeights/{config.modelSaveNameY}.h5")

modelU = upscalingModel.model1(upscaleFactor)
modelU.load_weights(f"./modelWeights/{config.modelSaveNameU}.h5")

modelV = upscalingModel.model1(upscaleFactor)
modelV.load_weights(f"./modelWeights/{config.modelSaveNameV}.h5")

inputSizeX = targetSizeX // upscaleFactor
inputSizeY = targetSizeY // upscaleFactor

testPath = config.loadTestImage
savePath = config.saveOutputImage


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor), PIL.Image.BICUBIC)


def upscale_image(modelY, modelU, modelV, img):
    """Predict the result based on input image and restore the image as RGB."""
    yuv = img.convert("YCbCr")
    y, u, v = yuv.split()

    y = img_to_array(y)
    y = y.astype("float16") / 255.0

    u = img_to_array(u)
    u = u.astype("float16") / 255.0

    v = img_to_array(v)
    v = v.astype("float16") / 255.0

    inputY = np.expand_dims(y, axis=0)
    inputU = np.expand_dims(u, axis=0)
    inputV = np.expand_dims(v, axis=0)

    outY = modelY.predict(inputY)
    outU = modelU.predict(inputU)
    outV = modelV.predict(inputV)

    out_img_y = outY[0]
    out_img_y *= 255.0

    out_img_u = outU[0]
    out_img_u *= 255.0

    out_img_v = outV[0]
    out_img_v *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")

    # Restore the image in RGB color space.
    out_img_u = out_img_u.clip(0, 255)
    out_img_u = out_img_u.reshape((np.shape(out_img_u)[0], np.shape(out_img_u)[1]))
    out_img_u = PIL.Image.fromarray(np.uint8(out_img_u), mode="L")

    # Restore the image in RGB color space.
    out_img_v = out_img_v.clip(0, 255)
    out_img_v = out_img_v.reshape((np.shape(out_img_v)[0], np.shape(out_img_v)[1]))
    out_img_v = PIL.Image.fromarray(np.uint8(out_img_v), mode="L")

    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_u, out_img_v)).convert("RGB")
    return out_img


# Loading All Test Images
listOftestImgs = []

for img in os.listdir(testPath):
    listOftestImgs.append(os.path.join(testPath, img))

testImages = sorted(listOftestImgs)

counter = 0
for index, test_img_path in enumerate(testImages):
    img = load_img(test_img_path)

    lowres_input = get_lowres_image(img, upscaleFactor)
    w = lowres_input.size[0] * upscaleFactor
    h = lowres_input.size[1] * upscaleFactor

    highres_img = img.resize((w, h))

    t0 = time.time()
    prediction = upscale_image(modelY, modelU, modelV, lowres_input)
    t1 = time.time()
    total = t1 - t0
    print(total)

    bicubic_img = lowres_input.resize((w, h))
    bicubic_img_arr = img_to_array(bicubic_img)

    highres_img_arr = img_to_array(highres_img)

    predict_img_arr = img_to_array(prediction)

    bicubic_psnr = PSNR(highres_img_arr, bicubic_img_arr)
    test_psnr = PSNR(highres_img_arr, predict_img_arr)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(f"PSNR of low resolution image and high resolution image is {bicubic_psnr}")
    print(f"PSNR of predict and high resolution is {test_psnr}")

    # plot_results(lowres_img, index, "lowres")
    # plot_results(highres_img, index, "highres")
    # plot_results(prediction, index, "prediction")
    prediction.save(f"{savePath}/aiScaled_{counter}.jpg")
    lowres_input.save(f"{savePath}/c_lowresInput_{counter}.jpg")
    highres_img.save(f"{savePath}/bicubicScaling_{counter}.jpg")

    counter += 1
