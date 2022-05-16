# Target Size For Training
targetSize = 256

# How Many Times To Scale Image
scaleFactor = 2

usingDataGen = True

# MAKE SURE TO CHANGE TO VAL AFTER
augmentSavePath = "./data/val/lowRes"
loadOriginal = "./data/val/original"

# Load Model
epochs = 100

# Original Model Name = 8k_2x

modelSaveName = "8k_2x_model2"

modelLoadName = "8k_2x_model2"

# Folder For Test Images
upscaleImageLocPath = "./data/testPexel/highRes"

# Folder For Output
upscaleImageSavePath = "./data/runs"


"""
How To Use

Start DataGen.Py To Generate Low Res Versions Of Data For Training
Also adds data augmentation
Make Sure To Enable usingDataGen and Disable After Using It To Prevent Errors
Make Sure To Switch To Val and Train Accordingly

Then Run dataLoader.py to save data to .h5 file so it can be access for model fitting

Then Run main.py to train data and save model weights, weights file name is above

upscalingModel.py contains the models used for upscaling, nothing to touch unless creating new model

upscaleImage.py is used for upscaling test images to measure performance, make sure to set weights file name correctly and weights match with model

Other Notes/ TODO
-Make Everything Cleaner
-Command Interface
-Better Performance
-Remove Objects or People From Image
-Simply Execution System Instead Of Running Multiple Files
-More Data
-Better Augmentation
-Other Implementations/using custom models lstm's, srgans etc.

"""