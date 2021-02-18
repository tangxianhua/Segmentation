import glob
import os
import cv2
import numpy as np
import pydicom
import pylab
from PIL import Image

dcmpath = r'C:\Users\fafafa\Desktop\0309\RG\dcm'
pngpath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\result7'
bdcmsavepath = r'C:\Users\fafafa\Desktop\0309\Compare\denseunet\test\merge\resultdcm7\\'
def myrgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = g

    return gray
for size in range(0,32):
    if size <= 9 :
      ds = pydicom.read_file(os.path.join(dcmpath,'0'+str(size)+'.dcm'))
      img = cv2.imread(os.path.join(pngpath,'0'+str(size)+'.png'))
    else:
      ds = pydicom.read_file(os.path.join(dcmpath, str(size) + '.dcm'))
      img = cv2.imread(os.path.join(pngpath,  str(size) + '.png'))
    print("开始读取：", ds.PatientName)
    img = myrgb2gray(img)
    matrix = np.asarray(img)

    arr = matrix.flatten()
    for i in range(262144):
        if arr[i] >= 50:
            ds.pixel_array.flat[i] = 65535
        else:
            ds.pixel_array.flat[i] = 0
    ds.PixelData = ds.pixel_array.tostring()
    if size <=9:
      ds.save_as(bdcmsavepath + '0'+str(size) + ".dcm")
    else:
      ds.save_as(bdcmsavepath + str(size) + ".dcm")
    print("开始存入：", ds.PatientName)

