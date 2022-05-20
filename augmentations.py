import tensorflow as tf


# Image Distortions and Augmentation
def jpgDist(img, minVal, maxVal):
    return tf.image.random_jpeg_quality(img, minVal, maxVal)
    # return tf.image.adjust_jpeg_quality(img, random.randint(99, 100))


# Higher Sigma = More Blur
# https://stackoverflow.com/a/59286930/11521629
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, sigma):
    blur = _gaussian_kernel(3, sigma, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def randomBlur(img, sigma):
    return apply_blur(img, sigma)


def downSample(img, lowResX, lowResY, scalingMethod):
    return tf.image.resize(img, (lowResX, lowResY), method=scalingMethod)


def upSample(img, fullResX, fullResY, scalingMethod):
    return tf.image.resize(img, (fullResX, fullResY), method=scalingMethod)


# https://stackoverflow.com/a/30609854/11521629
def randomNoise(img, noise_typ):
    shape = img.shape

    # Adding Gaussian noise
    noise = tf.random_normal(shape=shape, mean=0.0, stddev=noise_typ, dtype=tf.float32)
    return img + noise
