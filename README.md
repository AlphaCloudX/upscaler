# upscaler_ver2

This is based on the first ai model I did just changing the model to the one used by srcnn in the research paper:
https://arxiv.org/pdf/1808.03344.pdf

added file for augmenting data such as adding blur, noise, jpg distortion, downsampling, upsampling etc.

uses the same backend for processing data as ver_1 and still on tensorflow 2.8

using this model architecture solved artifacting and created a substantial improvement in image quality

this dataset was also trained on gameplay videos split into frames

loaded the weights from the new model and upscaled using the ver_1 code
