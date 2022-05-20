"""
Config File For Training And Running Upscaler
"""

# Size For The Image Array To Train On
trainingSizeX = 300
trainingSizeY = 300
upscalingFactor = 2

# Leave Blank For Default Values
# Directory Of Dataset
# your/file/location/data
dataDir = ""

# Specify Whether To Augment Data Or Not
doAugmentations = False

# Save Augmented Images So They Can Be Visually Inspected
saveAugImage = True

# Where To Save Augmented Images
# Leave Blank For Default Path
saveAugImagePath = ""

batchSize = 32
epochs = 2500
modelSaveNameY = "modelWeightsY"
modelSaveNameU = "modelWeightsU"
modelSaveNameV = "modelWeightsV"

loadModelName = "modelWeights"

loadTestImage = "./data/test"
saveOutputImage = "./data/runs"

# Upscale Single Image
image = "./data/test/landscape-nature-wallpaper-preview.jpg"
name = "landscape-nature-wallpaper-preview.jpg"
savePath = "./data/upscaledImages"
