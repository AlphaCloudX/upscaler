# upscaler

training data:
url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

peaks at 29gb of ram usage

trained for x2 upscaling 720p to 2k
 -2560x1440 <--> 1280x720

Progress:
Mon-Tues: researching what archecture to use, settled on cnn
planning on doing lstm or gan but too complex

wednesday: found source to work off of, source was broken/not functional so found another that was slightly different with its implementation

thursday reverse engineering the source and seeing the work flow of the project

friday and weekend and monday was coding from scratch with custom model(using modified mobilenetv2 architecture to benefit from faster upscale times)
