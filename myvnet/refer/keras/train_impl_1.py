import sys

from numpy import shape

sys.path.append('/export/software/tensorflow-1.3.0-rc2/python_modules/')

import numpy
import warnings
from keras.layers import Conv3D, Input, RepeatVector, Activation, concatenate, add, Conv3DTranspose, BatchNormalization, Dropout
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import activations, initializers, regularizers
from keras.engine import Layer, InputSpec
from keras.utils.conv_utils import conv_output_length
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.engine.topology import Layer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import functools
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
import os
import matplotlib
from keras.preprocessing import image

print(tf.__version__)

with open('/home/fafafa/KERAS/DATA/DATA50/mhd/train_data.p3', 'rb') as f:
    X, y = pickle.load(f)

X = X.reshape(X.shape + (1,)).astype(numpy.float32)
y = y.reshape(y.shape + (1,)).astype(numpy.float32)

print(X.shape)
print(y.shape)

def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4

_handle_data_format()
print('CHANNEL_AXIS = {}'.format(CHANNEL_AXIS))

from keras import backend as K


def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer
    for _ in range(n_convolutions-1):
        inl = PReLU(shared_axes=[1, 2, 3])(
            Conv3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format='channels_last')(inl)
        )
        inl = BatchNormalization(axis = CHANNEL_AXIS)(inl)
        inl = Dropout(0.5)(inl)

    conv = Conv3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format='channels_last')(inl)
    added = add([conv, input_layer])
    downsample = Conv3D(n_output_channels, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(added)
    prelu = PReLU(shared_axes=[1, 2, 3])(downsample)
    return prelu, added

def upward_layer(input0 ,input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=4)
    inl = merged
    for _ in range(n_convolutions-1):
        inl = PReLU(shared_axes=[1, 2, 3])(
            Conv3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format='channels_last')(inl)
        )
        inl = BatchNormalization(axis = CHANNEL_AXIS)(inl)
        inl = Dropout(0.5)(inl)
    conv = Conv3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format='channels_last')(inl)
    added = add([conv, merged])
    upsample = Conv3DTranspose(n_output_channels, (2, 2, 2), strides=(2 ,2, 2), padding='SAME', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', data_format='channels_last')(added)
    return PReLU(shared_axes=[1, 2, 3])(upsample), merged, inl, added



# Layer 1
input_layer = Input(shape=(256, 256, 32, 1), name='data')

conv_1 = Conv3D(8, (5, 5, 5), padding='same', data_format='channels_last')(input_layer)

repeat_1 = concatenate([input_layer] * 8)
add_1 = add([conv_1, repeat_1])
prelu_1_1 = PReLU()(add_1)
# downsample_1 = downward_layer(prelu_1_1, 2, 32)
downsample_1 = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(prelu_1_1)
prelu_1_2 = PReLU(shared_axes=[1, 2, 3])(downsample_1)

# Layer 2,3,4
out2, left2 = downward_layer(prelu_1_2, 2, 32)
out3, left3 = downward_layer(out2, 2, 64)
out4, left4 = downward_layer(out3, 2, 128)

# Layer 5
conv_5_1 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(out4)
prelu_5_1 = PReLU(shared_axes=[1, 2, 3])(conv_5_1)
conv_5_2 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(prelu_5_1)
prelu_5_2 = PReLU(shared_axes=[1, 2, 3])(conv_5_2)
conv_5_3 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(prelu_5_2)
add_5 = add([conv_5_3, out4])
prelu_5_1 = PReLU(shared_axes=[1, 2, 3])(add_5)#(16,16,2,256)
downsample_5 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='SAME', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', data_format='channels_last')(prelu_5_1)
prelu_5_2 = PReLU(shared_axes=[1, 2, 3])(downsample_5)#((?,?,?,?,128))

#Layer 6,7,8
out6, merged6, inl6, added6 = upward_layer(prelu_5_2, left4, 3, 32)
out7, merged7, inl7, added7 = upward_layer(out6, left3, 3, 16)
out8, merged8, inl8, added8 = upward_layer(out7, left2, 2, 8)

#Layer 9
merged_9 = concatenate([out8, add_1], axis=4)
conv_9_1 = Conv3D(16, (5, 5, 5), padding='same', data_format='channels_last')(merged_9)
add_9 = add([conv_9_1, merged_9])
conv_9_2 = Conv3D(1, (1, 1, 1), padding='same', data_format='channels_last', activation='sigmoid')(add_9)

print(conv_9_2)

model = Model(input_layer, conv_9_2)


def dice_coef(y_true, y_pred, smooth = 1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# def dice_coef(y_true, y_pred, smooth = 1e-5):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true) + K.sum(y_pred)
#     return (2. * intersection + smooth) / (union + smooth)
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

#smooth = 1. # 用于防止分母为0.
#def dice_coef(y_true, y_pred):
#       y_true_f = K.flatten(y_true) # 将 y_true 拉伸为一维.
#       y_pred_f = K.flatten(y_pred)
#       intersection = K.sum(y_true_f * y_pred_f)
#       return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
#def dice_coef_loss(y_true, y_pred):
#    return 1. - dice_coef(y_true, y_pred)


print('compile')
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

t=time.time()
print('test predict')
# y_pred = model.predict(X[:1,:,:,:,:])
print(time.time() - t)
model_checkpoint = ModelCheckpoint('kreas.hdf5', monitor='loss', save_best_only=True)
print(model.summary())
hist=model.fit(X, y, batch_size=1, epochs=10, verbose=1,callbacks = [model_checkpoint])

with open('keras.txt','w') as f:
    f.write(str(hist.history))
lossy = hist.history['loss']
plt.plot(lossy)
plt.savefig("keras.png")
