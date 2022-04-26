# upscaler

training data:
url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

peaks at 29gb of ram usage

trained for x2 upscaling 720p to 2k
 -2560x1440 <--> 1280x720

Progress:
Mon-Tues: researching what archecture to use, settled on cnn approach because have most experience
planning on doing lstm or gan but too complex

wednesday: found source to work off of, source was broken/not functional so found another that was slightly different with its implementation

thursday reverse engineering the source and seeing the work flow of the project, testing around how everything works

friday and weekend and monday was coding from scratch with custom model(using modified mobilenetv2 architecture to benefit from faster upscale times)
has shown to be effective but problem with accuracy

plans are to improve model, add more augmentations such as blur, jpg compression, noise, etc. <--tensorflow can't do these by batches
curently only using random vertical flip on data

improve speed, try to load model in c++, java or c-python
try to upscale low res image using regular upscaling algorithm and then working from there reducing time the ai has to compute

higher the psnr the better
also check against ssim for structural similarity index
