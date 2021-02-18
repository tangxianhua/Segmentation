
import sys
sys.path.append('/export/software/tensorflow-1.3.0-rc2/python_modules/')

from keras.models import load_model
import keras.backend as K
from keras.layers import Conv3D
from os import listdir
import SimpleITK
import os.path
import pickle
import numpy
import tensorflow as tf
import ntpath
import matplotlib.pyplot as plt


volSize = numpy.array((256,256,32), numpy.int32)
normDir = False
method = SimpleITK.sitkLinear


def generate_segmentation(f, model, output_path):
    print(f)
    scan = SimpleITK.ReadImage(f)
    ret = numpy.zeros(volSize, dtype=numpy.float32)

    newSize = numpy.array([256, 256, 32])
    factor = numpy.asarray(scan.GetSize()) / newSize
    new_spacing = numpy.asarray(scan.GetSpacing()) * factor

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(newSize.tolist())#????????
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)#????????

    imgResampled = resampler.Execute(scan)


    imgCentroid = numpy.asarray(newSize, dtype=numpy.float) / 2.0

    imgStartPx = (imgCentroid - numpy.array(volSize) / 2.0).astype(dtype=int)

    regionExtractor = SimpleITK.RegionOfInterestImageFilter()
    regionExtractor.SetSize(volSize.astype(dtype=numpy.int32).tolist())
    regionExtractor.SetIndex(imgStartPx.tolist())#??????

    imgResampledCropped = regionExtractor.Execute(imgResampled)

    X = SimpleITK.GetArrayFromImage(imgResampledCropped).astype(dtype=numpy.float)
    print(X.shape)

    X =  numpy.array( [numpy.transpose(X, [2, 1, 0])] )
    print(X.shape)

    X = X.reshape(X.shape + (1,)).astype(numpy.float32)
    y_pred = model.predict(X)

    y_pred = y_pred[0]
    y_pred=numpy.transpose(y_pred,[3,0,1,2])
    y_pred = numpy.transpose(y_pred[0], [2, 1, 0])

    import cv2
    for c in range(32):
        img=y_pred[c]
        img[img==1]=255
        img2 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        if c<=9:
            cv2.imwrite("0{}.png".format(c), img2)
        else:
            cv2.imwrite("{}.png".format(c),img2)
        print("{}.png".format(c))


    print(y_pred.shape)
    y_three=numpy.reshape(y_pred,(256,256,32))
    pred = numpy.zeros(imgResampled.GetSize(), dtype=numpy.float)
    print(pred.shape)

    pred[ 0:y_three.shape[0], 0:y_three.shape[1], 0:y_three.shape[2] ] = y_three#?????
    pred = numpy.transpose(pred, [2, 1, 0])


    mask = SimpleITK.GetImageFromArray(pred)
    mask.SetOrigin( scan.GetOrigin() )
    mask.SetDirection( scan.GetDirection() )
    mask.SetSpacing( scan.GetSpacing() )
    print(mask.GetSize())
    newSize3 = numpy.array([2048, 2048, 2048])
    newSize2 = numpy.array([256, 256, 32])
    factor = newSize2 / newSize3
    new_spacing2 = numpy.asarray(mask.GetSpacing()) * factor


    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(mask)
    resampler.SetOutputSpacing(new_spacing2)
    resampler.SetSize(newSize3.tolist())
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)

    imgReverseResampled = resampler.Execute(mask)

    print(scan.GetSize())
    print(imgReverseResampled.GetSize())

    if not output_path.endswith('/'):
        output_path = output_path + '/'

    filename = ntpath.basename(f)
    SimpleITK.WriteImage(SimpleITK.Cast(imgReverseResampled, SimpleITK.sitkUInt8), output_path+filename[:-4]+'ce2048.mhd')

    # print('-----------')

def iterate_folder(folder):
    for filename in sorted(listdir(folder)):
        if filename.endswith('.mhd') and not filename.endswith('_segmentation.mhd'):
            absolute_filename = os.path.join(folder, filename)
            yield absolute_filename
#
def dice_coef(y_true, y_pred, smooth = 1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
    #return -dice_coef(y_true, y_pred)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def process_data(folder):
    input_filenames = list(iterate_folder(folder))
    model = load_model(r'/home/fafafa/KERAS/result_VNET/data50/incept3-1227-500/incept3-1227-500.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 'Conv3D':Conv3D,'tf':tf})
    #model = load_model(r'/home/fafafa/vnet_0719keras-master/result_VNET/vnet-1119-500-dicenew/vnet-1119-500-dicenew.hdf5')
    for f in input_filenames:
        generate_segmentation(f, model, r'/home/fafafa/KERAS/result_VNET//')
process_data(r'/home/fafafa/KERAS/TEST/256')

