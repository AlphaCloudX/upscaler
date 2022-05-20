# Upscaler Version 4 -Final Version

Redid custom data loader to be more efficient when loading data

-now supports multithreading of validation and training data at the same time

-also writes the high res and low res data to the array at the same time and appends each list at the end

Before bicubic upscaling was used to upsample U and V layers of image, changed it so that the Ai does those aswell making it a full ai implementation, no other upscaling method is used

-3 models for Y,U,V channels then image is merged into one

Still using Sr-SLNN modified implementation as it works well enough
Goal was to do 3d convolutions with 3 channel image but couldn't get it to work so settled for just upsampling 3 channels individually

Added extra file to upscale single image to the new dimension instead of testing how it performs.

Config file has been cleaned up with more meaningful parameters

File structure of the project is more clean with data being devided based on validation, test, train, runs(from tests) and upscaledImages which is the upscaled output of the single image

Data folder is now:
data/runs

data/test

data/training/
data/training/augmented
data/training/original

data/upscaledImages

data/validation
data/validation/augmented
data/validation/original

data/imageDatah5

data/modelWeights

imageDatah5 contains all the h5 files for the different channels for val and train data

modelWeights folder contains the 3 weights used by the model, allows model to be fine tuned to specific channels

-Also means that the code can be edited to use other data formats such as RGB and then merge them

Goal was to beat bicubic upsampling using a pure ai approach which it is capable of doing as seen in the examples.
