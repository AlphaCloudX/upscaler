import PIL
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import time
from PIL import Image

import upscalingModel
import config

image = config.image
name = config.name
savePath = config.savePath
upscaleFactor = config.upscalingFactor

# Most use x2 or x4 for 300

modelY = upscalingModel.model1(upscaleFactor)
modelY.load_weights(f"./modelWeights/{config.modelSaveNameY}.h5")

modelU = upscalingModel.model1(upscaleFactor)
modelU.load_weights(f"./modelWeights/{config.modelSaveNameU}.h5")

modelV = upscalingModel.model1(upscaleFactor)
modelV.load_weights(f"./modelWeights/{config.modelSaveNameV}.h5")


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


img = load_img(image)

t0 = time.time()
prediction = upscale_image(modelY, modelU, modelV, img)
t1 = time.time()
total = t1 - t0
print(f"time taken: {total}sec")

bicubic_img = img.resize((img.size[0] * upscaleFactor, img.size[1] * upscaleFactor))
bicubic_img_arr = img_to_array(bicubic_img)

predict_img_arr = img_to_array(prediction)

prediction.save(f"{savePath}/aiScaled_{name}")
bicubic_img.save(f"{savePath}/bicubicScaling_{name}")
