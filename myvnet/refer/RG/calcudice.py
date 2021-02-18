import glob
import os
import cv2
import numpy as np
import pydicom
import tensorflow.keras.backend as K
from keras import backend as K
import tensorflow as tf



predpath = r'C:\Users\fafafa\Desktop\wuda\result\rgbindary'
prelist = glob.glob(os.path.join(predpath,'*.dcm'))
prelist.sort()
gtpath = r'C:\Users\fafafa\Desktop\wuda\result\8bindaryGTdcm'
gtlist = glob.glob(os.path.join(gtpath,'*dcm'))
gtlist.sort()

allresult = 0


def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)
for j in range(32):
    gt = pydicom.read_file(gtlist[j])
    gtzero = np.zeros(shape=(512, 512))
    for i in range(262144):
        gtzero.flat[i] = gt.pixel_array.flat[i]
    gttensor = K.constant(gtzero)
    pred = pydicom.read_file(prelist[j])
    predzero = np.zeros(shape=(512, 512))
    for i in range(262144):
        predzero.flat[i] = pred.pixel_array.flat[i]
    predtensor = K.constant(predzero)

    diceresult = dice_coef(gttensor, predtensor)
    print('dice',diceresult)
    sess = tf.Session()
    result = sess.run(diceresult)
    trueresult = (result / 65535)
    allresult += trueresult
    print('allresult',allresult)
allresultmean = allresult / 32
print(allresultmean)
