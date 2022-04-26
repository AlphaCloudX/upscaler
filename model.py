import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add, Softmax, AveragePooling2D, \
    Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow import keras


# https://medium.com/analytics-vidhya/creating-mobilenetsv2-with-tensorflow-from-scratch-c85eb8605342

def expansion_block(x, t, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t * filters
    x = Conv2D(total_filters, 1, padding='same', use_bias=False, name=prefix + 'expand')(x)
    x = BatchNormalization(name=prefix + 'expand_bn')(x)
    x = ReLU(6, name=prefix + 'expand_relu')(x)
    return x


def depthwise_block(x, stride, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3, strides=(stride, stride), padding='same', use_bias=False, name=prefix + 'depthwise_conv')(x)
    x = BatchNormalization(name=prefix + 'dw_bn')(x)
    x = ReLU(6, name=prefix + 'dw_relu')(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters=out_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + 'compress')(x)
    x = BatchNormalization(name=prefix + 'compress_bn')(x)
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


def model1(upscaleFactor):
    input = keras.Input(shape=(None, None, 1))
    x = Conv2D(64, 5, **conv_args)(input)

    # x = Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(input)
    # x = BatchNormalization(name='conv1_bn')(x)
    # x = ReLU(6, name='conv1_relu')(x)
    # 17 Bottlenecks

    # Use Stride 1 to prevent error with image dimensions
    # ValueError: Dimensions must be equal, but are 150 and 300 for '{{node mean_squared_error/SquaredDifference}} = SquaredDifference[T=DT_FLOAT](model/tf.nn.depth_to_space/DepthToSpace, IteratorGetNext:1)' with input shapes: [?,150,150,1], [?,300,300,1].
    # Output is suppose to be 300x300x1(channel) but using higher stride causes the dimensions to decrease by factor

    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=32, block_id=1)

    # x = 32 Filters Incoming
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=2)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=3)

    # Commented Out For Performance Reasons
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=4)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=5)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=6)
    #
    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=7)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=8)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=9)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=10)

    x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=11)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=12)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=13)

    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=14)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=15)
    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=16)

    #x = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=320, stride=1, block_id=17)

    # Keep this out
    # 1*1 conv
    # x = Conv2D(filters=1280, kernel_size=1, padding='same', use_bias=False, name='last_conv')(x)

    # x = BatchNormalization(name='last_bn')(x)
    # x = ReLU(6, name='last_relu')(x)

    x = Conv2D(1 * (upscaleFactor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscaleFactor)

    model = Model(input, outputs)

    return model
