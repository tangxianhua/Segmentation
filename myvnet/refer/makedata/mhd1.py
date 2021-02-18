import glob
import os
import cv2
import numpy as np
import pydicom
import pylab
from PIL import Image
name1 = 'I:\changedata\standtrain'
for xunhuan in range(1,51):
    if xunhuan <= 9:
        name2 = '0'+str(xunhuan)
    else:
        name2 = str(xunhuan)
    WSI_MASK_PATH = os.path.join(name1,name2,'origin2part')
    #WSI_MASK_PATH = r'I:\changedata\Test\02\origin2part'
    paths_png = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
    paths_dcm = glob.glob(os.path.join(WSI_MASK_PATH, '*.dcm'))
    paths_png.sort()
    paths_dcm.sort()
    print(paths_png)
    print(paths_dcm)
    fragments = WSI_MASK_PATH.split('\\')
    fragmentspath = os.path.join(fragments[0], fragments[1], fragments[2], fragments[3])
    bpngpathname = 'bpng\\'
    print(bpngpathname)
    bpngpath = os.path.join(fragmentspath,bpngpathname)
    corppathname = 'corp\\'
    print(corppathname)
    corppath = os.path.join(fragmentspath,corppathname)
    j = 0

    for size in range(len(paths_dcm)):
        ds = pydicom.read_file(paths_dcm[size])
        print("开始读取：", ds.PatientName)
        img = cv2.imread(paths_png[size])
        Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(Grayimg, 30, 255, cv2.THRESH_BINARY)
        if j <=9:
          cv2.imwrite(bpngpath+ '0'+str(j) + ".png", thresh, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
          img = Image.open(bpngpath + '0'+str(j) + ".png")
        else:
          cv2.imwrite(bpngpath + str(j) + ".png", thresh, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
          img = Image.open(bpngpath + str(j) + ".png")


        matrix = np.asarray(img)
        arr = matrix.flatten()
        for i in range(262144):
            if arr[i] < 100:
                ds.pixel_array.flat[i] = 0
        ds.PixelData = ds.pixel_array.tostring()
        if j<=9:
          ds.save_as(corppath + '0'+str(j) + ".dcm")
        else:
          ds.save_as(corppath + str(j) + ".dcm")
        print("开始存入：", ds.PatientName)
        j = j + 1
        print(j)


