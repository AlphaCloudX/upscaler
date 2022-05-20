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

Original:

![landscape-nature-wallpaper-preview](https://user-images.githubusercontent.com/66267343/169445354-d6f61094-7951-488e-9b0b-e39377079f27.jpg)


Ai Scaled:
![aiScaled_landscape-nature-wallpaper-preview](https://user-images.githubusercontent.com/66267343/169442279-33419925-34e6-484c-94cd-124e9b733c53.jpg)


Bicubic:
![bicubicScaling_landscape-nature-wallpaper-preview](https://user-images.githubusercontent.com/66267343/169442280-d37790e4-c917-4bfc-ab28-3dd784fb1288.jpg)

Bicubic is more pixelated with the trees in particular being more noisy
The Ai Scaled version does look a bit more washed out with much more smoothing which can work for or against it's upsampling

From afar bicubic looks better because of the added sharpness in the image

Close Up of mountain
![image](https://user-images.githubusercontent.com/66267343/169442576-930bcb2e-c3c8-400c-8ccd-035ae8f38917.png)

The left image is the bicubic and the right is the ai scaled version,
this was performed on a 728x486 image and upsampled to 1456x972, by a factor of x2

Bicubic has advantage of looking sharper but when zoomed in on quickly loses quality, the ai can further be fine tuned by adding in post processing effects such as sharpening.

We see that smoothing out the image in this case works to our favour as seen in the water tide
![image](https://user-images.githubusercontent.com/66267343/169444996-3ee4175f-3cda-4634-8108-7f72d15bb66e.png)

the bicubc version is adding too much noise that doesn't look right, this is also seen on the tree stumps

----------

Stuff I Still Want To Implement:
using kneat in a gan style appraoch where 1 kneat algorithm generates an image, another kneat algorithm guesses on which one is fake and which one is real and keep going until they look near similar, can take advantage of bigger population sizes and seeing if neuroevoltion can helpout with creating better images as it learns what works and what doesn't

generating lots of lower resolution shapes with colour patterns that are commonly found in images in hopes ai can train on shape upscaling and use that knowledge to upsample similar shapes in the image. For example using low resolution rectangles to better upsample images with windows or rectangular signs. Or using triangles for more complex shapes.
