import h5py
from tensorflow import keras

import config
import upscalingModel

trainHighResPath = "./highResTrain.h5"
trainLowResPath = "./lowResTrain.h5"

valHighResPath = "./highResVal.h5"
valLowResPath = "./lowResVal.h5"

upscaleFactor = config.upscalingFactor

targetSizeX = config.trainingSizeX
targetSizeY = config.trainingSizeY


def readFile(file, targetSizeX, targetSizeY):
    f = h5py.File(file, 'r')
    data = f['data'][...]
    f.close()

    data = data.reshape((len(data)), targetSizeX, targetSizeY, 1)
    return data


trainX_Y = readFile("./imageDatah5/lowResTrain_dataLowY.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)
trainX_U = readFile("./imageDatah5/lowResTrain_dataLowU.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)
trainX_V = readFile("./imageDatah5/lowResTrain_dataLowV.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)

trainY_Y = readFile("./imageDatah5/highResTrain_dataHighY.h5", targetSizeX, targetSizeY)
trainY_U = readFile("./imageDatah5/highResTrain_dataHighU.h5", targetSizeX, targetSizeY)
trainY_V = readFile("./imageDatah5/highResTrain_dataHighV.h5", targetSizeX, targetSizeY)

valX_Y = readFile("./imageDatah5/lowResVal_dataLowY.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)
valX_U = readFile("./imageDatah5/lowResVal_dataLowU.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)
valX_V = readFile("./imageDatah5/lowResVal_dataLowV.h5", targetSizeX // upscaleFactor, targetSizeY // upscaleFactor)

valY_Y = readFile("./imageDatah5/highResVal_dataHighY.h5", targetSizeX, targetSizeY)
valY_U = readFile("./imageDatah5/highResVal_dataHighU.h5", targetSizeX, targetSizeY)
valY_V = readFile("./imageDatah5/highResVal_dataHighV.h5", targetSizeX, targetSizeY)

print("Starting Model")

loss_function = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
callbacks = [early_stopping_callback]

model = upscalingModel.model1(upscaleFactor)

model.compile(optimizer=optimizer, loss=loss_function)

# Train Y
model.fit(trainX_Y, trainY_Y, epochs=config.epochs, validation_data=(valX_Y, valY_Y),
          verbose=1, batch_size=config.batchSize)

model.save(f"./modelWeights/{config.modelSaveNameY}.h5")

# Train U
model.fit(trainX_U, trainY_U, epochs=config.epochs, validation_data=(valX_U, valY_U),
          verbose=1, batch_size=config.batchSize)

model.save(f"./modelWeights/{config.modelSaveNameU}.h5")

# Train V
model.fit(trainX_V, trainY_V, epochs=config.epochs, validation_data=(valX_V, valY_V),
          verbose=1, batch_size=config.batchSize)

model.save(f"./modelWeights/{config.modelSaveNameV}.h5")
