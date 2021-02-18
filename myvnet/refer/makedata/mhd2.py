import glob
import os
import cv2
import numpy as np
import pydicom
import pylab

name1 = 'I:\changedata\Test'
for i in range(1,11):
    if i <= 9:
        name2 = '0'+str(i)
    else:
        name2 = str(i)
    WSI_MASK_PATH = os.path.join(name1,name2,'corp')
    #WSI_MASK_PATH = r'I:\changedata\Test\10\corp'

    paths_dcm = glob.glob(os.path.join(WSI_MASK_PATH, '*.dcm'))
    paths_dcm.sort()
    print(paths_dcm)
    fragments = WSI_MASK_PATH.split('\\')
    fragmentspath = os.path.join(fragments[0], fragments[1], fragments[2], fragments[3])
    bdcmpathname='65535\\'
    bdcmpath = os.path.join(fragmentspath,bdcmpathname)
    print(bdcmpath)
    j = 0
    for size in range(len(paths_dcm)):
        ds = pydicom.read_file(paths_dcm[size])
        print("开始读取：", ds.PatientName)
        for t in range(262144):
            if ds.pixel_array.flat[t] > 0:
               ds.pixel_array.flat[t] = 65535
        ds.PixelData = ds.pixel_array.tostring()
        if j<=9:
          ds.save_as(bdcmpath + '0'+str(j) + ".dcm")
        else:
            ds.save_as(bdcmpath + str(j) + ".dcm")
        print("开始存入：", ds.PatientName)
        j = j + 1
        print(j)
