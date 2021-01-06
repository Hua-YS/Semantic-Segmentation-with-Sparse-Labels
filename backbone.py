from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Lambda, Add, Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
#import tensorflow.compat.v1 as tf

def VGG16(patch_size, bn=False):

    img_input = Input(shape=(patch_size, patch_size, 3), name='input1')
    x = conv2d(img_input, 64, 3, 'same', 'block1_conv1', bn)
    x = conv2d(x, 64, 3, 'same', 'block1_conv2', bn)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv2d(x, 128, 3, 'same', 'block2_conv1', bn)
    x = conv2d(x, 128, 3, 'same', 'block2_conv2', bn)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv2d(x, 256, 3, 'same', 'block3_conv1', bn)
    x = conv2d(x, 256, 3, 'same', 'block3_conv2', bn)
    x = conv2d(x, 256, 3, 'same', 'block3_conv3', bn)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv2d(x, 512, 3, 'same', 'block4_conv1', bn)
    x = conv2d(x, 512, 3, 'same', 'block4_conv2', bn)
    x = conv2d(x, 512, 3, 'same', 'block4_conv3', bn)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv2d(x, 512, 3, 'same', 'block5_conv1', bn)
    x = conv2d(x, 512, 3, 'same', 'block5_conv2', bn)
    x = conv2d(x, 512, 3, 'same', 'block5_conv3', bn)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x, name='vgg16')

    return model

def conv2d(x, nb_filters, filter_size, padding, name, bn=False):
    x = Conv2D(nb_filters, (filter_size, filter_size), padding=padding, name=name)(x)
    if bn==True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

