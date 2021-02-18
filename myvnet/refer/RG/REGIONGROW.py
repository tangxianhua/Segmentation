import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import tensorflow.keras.backend as K
from keras import backend as K
import tensorflow as tf

N= 1
bindaryoriginpath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\65535'
findcontourdcmpath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\resultdcm7'
contourpngsavepath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\contourpng'
originpath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\dcm'

regiongrowsave = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\regiongrow\\'
contourpngsavepathsave = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\7\contourpng\\'
#originpathsave = r'/home/fafafa/vnet_0719keras-master/regiongrow/origindcm//'

for xunhuan in range (N):
    print(xunhuan)
    contourlist = glob.glob(os.path.join(findcontourdcmpath,'*.dcm'))
    contourlist.sort()
    for slice in range(32):
       beforecontour = pydicom.read_file(contourlist[slice])
       contourzero = np.zeros(shape=(512,512))
       for i in range(512):
            for j in range(512):
                if beforecontour.pixel_array[i][j] != 0:
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if (beforecontour.pixel_array[i + x][j + y] == 0) :
                                contourzero[i][j] = 255
       if slice<=9:#保存边缘png
         cv2.imwrite(contourpngsavepathsave +"0{}.png".format(slice),contourzero)
       else:
         cv2.imwrite(contourpngsavepathsave +"{}.png".format(slice),contourzero)

    originpath = originpath
    originlist = glob.glob(os.path.join(originpath, '*.dcm'))
    originlist.sort()
    contourpath = contourpngsavepath
    contourlist = glob.glob(os.path.join(contourpath, '*.png'))
    contourlist.sort()
    predpath = findcontourdcmpath
    predlist = glob.glob(os.path.join(predpath, '*.dcm'))
    predlist.sort()

    for slice in range(32):
        origin = pydicom.read_file(originlist[slice])
        pred = pydicom.read_file(predlist[slice])
        contour = cv2.imread(contourlist[slice], 0)
        matrixcontour = np.asarray(contour)
        for i in range(512):
            for j in range(512):
                if matrixcontour[i][j] != 0:
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if (origin.pixel_array[i + x][j + y] >= 100) & (origin.pixel_array[i + x][j + y] <= 300):
                                pred.pixel_array[i + x][j + y] = 65535
                                print('change', i + x, i + y, pred.pixel_array[i + x][j + y])
                                pred.PixelData = pred.pixel_array.tostring()
        if slice <= 9:#保存膨胀后的pre去原路经
            pred.save_as(regiongrowsave + "0{}.dcm".format(slice))
        else:
            pred.save_as(regiongrowsave + "{}.dcm".format(slice))

    gtpath = bindaryoriginpath
    gtpathlist = glob.glob(os.path.join(gtpath, '*.dcm'))
    gtpathlist.sort()
    predpath = findcontourdcmpath
    predlist = glob.glob(os.path.join(predpath, '*.dcm'))
    predlist.sort()

    allresult = 0
    for j in range(32):
        gt = pydicom.read_file(gtpathlist[j])
        gtzero = np.zeros(shape=(512, 512))
        for i in range(262144):
            gtzero.flat[i] = gt.pixel_array.flat[i]
        gttensor = K.constant(gtzero)
        pred = pydicom.read_file(predlist[j])
        predzero = np.zeros(shape=(512, 512))
        for i in range(262144):
            predzero.flat[i] = pred.pixel_array.flat[i]
        predtensor = K.constant(predzero)
        def dice_coef(y_true, y_pred, smooth=1e-5):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            union = K.sum(y_true) + K.sum(y_pred)
            return (2. * intersection + smooth) / (union + smooth)
        diceresult = dice_coef(gttensor, predtensor)
        sess = tf.Session()
        result = sess.run(diceresult)
        trueresult = (result / 65535)
        allresult += trueresult
    allresultmean = allresult / 32
    print(allresultmean)



