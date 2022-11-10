import numpy as np
from keras.models import *
from keras.layers import *
from nets.vgg16 import VGG16
from keras.layers import Dense,LeakyReLU


alpha = 1
def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)
    return x

def conv_block(inputs, filters, kernel, strides, nl):
    # 一个卷积单元，也就是conv2d + batchnormalization + activation
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)

def cSE(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x


def sSE(inputs):
    input_channels = int(inputs.shape[-1])
    x = Conv2D(input_channels, 1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    return x


def csSE(inputs):
    c = cSE(inputs)
    s = sSE(inputs)
    return Add()([c, s])

#def squeeze(inputs):
    # 注意力机制单元
    #input_channels = int(inputs.shape[-1])

    #x = GlobalAveragePooling2D()(inputs)
    #x = Dense(int(input_channels/4))(x)
    #x = Activation(relu6)(x)
    #x = Dense(input_channels)(x)
    #x = Activation(hard_swish)(x)


    #x = Reshape((1, 1, input_channels))(x)
    #x = Multiply()([inputs, x])

    #return x


def HDC(x):
    x = Conv2D(512, 3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 3, dilation_rate=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x
def newHDC(x):
    # input_channels = int(x.shape[-1])
    # print(input_channels)
    #通道1
    x1 = Conv2D(128, 3, dilation_rate=1, padding='same')(x)
    x1 = Conv2D(128, 3, dilation_rate=1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = Conv2D(128, 3, dilation_rate=3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = Conv2D(128, 3, dilation_rate=3, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = cSE(x1)
    #通道2
    x2 = Conv2D(128, 3, dilation_rate=1, padding='same')(x)
    x2 = Conv2D(128, 3, dilation_rate=1, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(128, 3, dilation_rate=2, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = Conv2D(128, 3, dilation_rate=3, padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = cSE(x2)
    #通道3
    x3 = Conv2D(128, 3, dilation_rate=1, padding='same')(x)
    x3 = Conv2D(128, 3, dilation_rate=1, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x3 = Conv2D(128, 3, dilation_rate=2, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x3 = Conv2D(128, 3, dilation_rate=5, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x3 = cSE(x3)
    #通道4
    x4 = Conv2D(128, 3, dilation_rate=1, padding='same')(x)
    res = Concatenate(axis=3)([x1, x2])
    res = Concatenate(axis=3)([res, x3])
    res = Concatenate(axis=3)([res, x4])
    return res
def Unet(input_shape=(256,256,3), num_classes=21):
    inputs = Input(input_shape)
    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs) 
    channels = [64, 128, 256, 512]
    # #   HDC1
    # feat5 = HDC(feat5)
    # #   HDC2
    # feat5 = HDC(feat5)
    # # #   HDC3
    # feat5 = HDC(feat5)
    feat5 = newHDC(feat5)
    P5_up = UpSampling2D(size=(2, 2))(feat5)
    P4 = Concatenate(axis=3)([feat4, P5_up])
    P4 = csSE(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)

    P4_up = UpSampling2D(size=(2, 2))(P4)
    P3 = Concatenate(axis=3)([feat3, P4_up])
    P3 = csSE(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)

    P3_up = UpSampling2D(size=(2, 2))(P3)
    P2 = Concatenate(axis=3)([feat2, P3_up])
    P2 = csSE(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)

    P2_up = UpSampling2D(size=(2, 2))(P2)
    P1 = Concatenate(axis=3)([feat1, P2_up])
    P1 = csSE(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    model = Model(inputs=inputs, outputs=P1)
    return model

