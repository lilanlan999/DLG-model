from __future__ import division
import keras

from keras.layers import Flatten,BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.models import Sequential

from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import six
from keras.regularizers import l2
from keras.models import Model

from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K

"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,DenseNet
"""

class MODEL(object):

    def __init__(self,config):
        self.config = config

    def input_shape_define(self):
        return  (self.config.normal_size, self.config.normal_size, self.config.channles)

    def covn_block(self,model,kenal_number,kenal_size,padding,activation):
        model.add(Convolution2D(kenal_number,kenal_size,padding=padding,activation=activation))
        return model

    def max_pooling_type(self,model,kenal_size,strides):
        model.add(MaxPooling2D(pool_size=kenal_size,strides=strides))
        return model

    def NEW(self):
        model = Sequential()
        input_shape = (3,776,776)
        model.add(
            Convolution2D(

                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='tf',
                input_shape=input_shape,

            )
        )
        model.add(BatchNormalization())

        model.add(Activation('relu'))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Dropout(0.15))

        model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Dropout(0.15))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        return model
