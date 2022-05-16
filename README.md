# upscaler_ver3

This version was completely overhauled for tf 1.15.5 to work with directml to take advantage of gpu processing

Added Completely custom data load, includes:
-Loading data and adding in custom augmentations and saving augmented images to files to be looked over

-dataloader to load highres and augmentated images into h5 file so it can be saved in single file without having to load all images, also converts to YUV format

-main.py reads the h5 file into a numpy array and shapes the data so it can be used, then instead of feeding the model tensors, it's fed numpy arrays that correspond to images

-model then trains from train/valX which is lowRes images, and the answer/expected output is train/valY

variables file which allows for variables to be changed on the fly and take effect in all files.

compareFromRuns file which allows benchmarking of images from different models to see which one works best on a per pixel basis

No general improvements to previous model performance other than adding 2nd option but this limitation may be because of only using single channel and not using 3 channeled rgb image

Now is easy to train with new data, modify model on the fly, modify parameters, and most importantly take advantage of directml to use any gpu to train ai model.

Data Format:

train

-lowRes

-original

val

-lowRes

-original

test

- highRes

# Speeds up time to 4seconds a batch of 1000 images at 256x256x1 compared to 174seconds for 400 images at 300x300x1 in previous versions


to get done

optimize data loading -many redundant for loops

improve model --> allowing 3 channels

clean up code
