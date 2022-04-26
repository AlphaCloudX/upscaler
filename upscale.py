import cv2
import PIL
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import time
import os
import tensorflow as tf

import model

total_bicubic_psnr = 0.0
total_test_psnr = 0.0


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor), PIL.Image.BICUBIC)


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert("RGB")
    return out_img


root_path = "./data/images"

targetSize = 300
upscaleFactor = 2
inputSize = targetSize // upscaleFactor

testPath = root_path + "/test"

# Loading All Test Images
listOftestImgs = []

for img in os.listdir(testPath):
    listOftestImgs.append(os.path.join(testPath, img))

testImages = sorted(listOftestImgs)

modell = model.model1(upscaleFactor)
modell.load_weights("weights.h5")

counter = 0
for index, test_img_path in enumerate(testImages[0:10]):
    t0 = time.time()
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscaleFactor)
    w = lowres_input.size[0] * upscaleFactor
    h = lowres_input.size[1] * upscaleFactor
    highres_img = img.resize((w, h))
    prediction = upscale_image(modell, lowres_input)
    t1 = time.time()
    total = t1 - t0
    print(total)

    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print("PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr)
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    # plot_results(lowres_img, index, "lowres")
    # plot_results(highres_img, index, "highres")
    # plot_results(prediction, index, "prediction")
    prediction.save(f"test{counter}.jpg")

    counter += 1
