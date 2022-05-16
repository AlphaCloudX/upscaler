import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add, Softmax, AveragePooling2D, \
    Dense, Input, GlobalAveragePooling2D


# https://medium.com/analytics-vidhya/creating-mobilenetsv2-with-tensorflow-from-scratch-c85eb8605342

def expansion_block(x, t, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t * filters
    x = Conv2D(total_filters, 1, padding='same', use_bias=False, name=prefix + 'expand')(x)
    # x = BatchNormalization(name=prefix + 'expand_bn')(x)
    # x = ReLU(6, name=prefix + 'expand_relu')(x)
    return x


def depthwise_block(x, stride, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3, strides=(stride, stride), padding='same', use_bias=False, name=prefix + 'depthwise_conv')(x)
    # x = BatchNormalization(name=prefix + 'dw_bn')(x)
    # x = ReLU(6, name=prefix + 'dw_relu')(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters=out_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + 'compress')(x)
    # x = BatchNormalization(name=prefix + 'compress_bn')(x)
    return x


def Bottleneck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = add([x, y])
    return y


conv_args = {"activation": "relu",
             "kernel_initializer": "Orthogonal",
             "padding": "same"}


# Using SR-SLNN from research paper with modified implementation
def model1(upscaleFactor):
    inputs = keras.Input(shape=(None, None, 1))  # None means we can take any input size and 1 is only 1 channel

    x = Conv2D(64, 5, **conv_args)(inputs)

    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(32, 1, **conv_args)(x)

    x = Conv2DTranspose(32, 4, **conv_args)(x)
    x = Conv2DTranspose(32, 4, **conv_args)(x)

    x = Conv2D(16, 3, **conv_args)(x)

    x = Conv2D(1 * (upscaleFactor ** 2), 3, **conv_args)(x)

    outputs = tf.nn.depth_to_space(x, upscaleFactor)

    model = Model(inputs, outputs)

    return model


def model2(upscaleFactor):

    inputs = keras.Input(shape=(None, None, 1))

    x = Conv2D(64, 3, **conv_args)(inputs)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)

    x = Conv2D(32, 1, **conv_args)(x)

    x = Conv2DTranspose(32, 4, **conv_args)(x)
    x = Conv2DTranspose(32, 3, **conv_args)(x)
    x = Conv2DTranspose(32, 4, **conv_args)(x)

    x = Conv2D(16, 3, **conv_args)(x)

    x = Conv2D(64, 5, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)
    x = Conv2D(64, 3, **conv_args)(x)

    x = Conv2D(16, 3, **conv_args)(x)

    x = Conv2D(32, 3, **conv_args)(x)

    x = Conv2D(1 * (upscaleFactor ** 2), 3, **conv_args)(x)

    outputs = tf.nn.depth_to_space(x, upscaleFactor)

    model = Model(inputs, outputs)

    return model
