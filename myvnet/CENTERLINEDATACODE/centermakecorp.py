import glob
import os
import cv2
import numpy as np
import pydicom
import pylab
from PIL import Image


path = 'I:\\maskcenterline'
centerline = 'maskcenterline'

dcm = 'dcm'
skeleton = 'skeleton'
corp = 'corp'
bdcm = 'bdcm'
j = 0

def myrgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = g

    return gray

for i in range(1,51):
    if i <= 9:
        dcmpath1 = os.path.join(path, '0' + str(i), centerline,dcm)
        skeletonpath1 = os.path.join(path, '0' + str(i), centerline,skeleton)
        corppath1 = os.path.join(path,'0' + str(i), centerline,corp)
        bdcmpath1 = os.path.join(path,'0' + str(i), centerline,bdcm)
    else:
        dcmpath1 = os.path.join(path,  str(i), centerline,dcm)
        skeletonpath1 = os.path.join(path,  str(i), centerline,skeleton)
        corppath1 = os.path.join(path, str(i), centerline,corp)
        bdcmpath1 = os.path.join(path, str(i), centerline, bdcm)
    for j in range (0,32):
        if j <= 9:
            dcmpath = os.path.join(dcmpath1,'0'+str(j)+'.dcm')
            skeletonpath = os.path.join(skeletonpath1, '0' + str(j) + '.png')
            savecorppath = os.path.join(corppath1,'0' + str(j))
            savebdcmpath = os.path.join(bdcmpath1,'0' + str(j))
        else:
            dcmpath = os.path.join(dcmpath1, str(j) + '.dcm')
            skeletonpath = os.path.join(skeletonpath1, str(j) + '.png')
            savecorppath = os.path.join(corppath1,  str(j))
            savebdcmpath = os.path.join(bdcmpath1,  str(j))
        print(dcmpath)
        print(skeletonpath)


        ds = pydicom.read_file(dcmpath)
        print("开始读取DCM：", ds.PatientName)
        img = cv2.imread(skeletonpath)
        print("开始读取PNG：", skeletonpath)

        img = myrgb2gray(img)
        matrix = np.asarray(img)

        arr = matrix.flatten()
        for z in range(262144):
              if arr[z] <100:
                ds.pixel_array.flat[z] = 0
        ds.PixelData = ds.pixel_array.tostring()
        print("开始存入：", ds.PatientName)
        ds.save_as(savecorppath + ".dcm")

        # print('开始存入',savecorppath+'.dcm')
        # print('读取裁剪dcm',savecorppath+'.dcm')


        print("开始读取裁剪DCM：", ds.PatientName)
        ds = pydicom.read_file(savecorppath+'.dcm')
        for t in range(262144):
            if ds.pixel_array.flat[t] > 0:
                ds.pixel_array.flat[t] = 65535
        ds.PixelData = ds.pixel_array.tostring()
        print("开始存入二值DCM：", ds.PatientName)
        ds.save_as(savebdcmpath + ".dcm")

        #print('开始存入二值DCM',savebdcmpath + ".dcm")





