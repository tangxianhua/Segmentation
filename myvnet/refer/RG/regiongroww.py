import glob
import os
import cv2
import numpy as np
import pydicom
cotourpngpath = r'C:\Users\fafafa\Desktop\wuda\result\8512contourimage'
contourlist = glob.glob(os.path.join(cotourpngpath,'*.png'))
contourlist.sort()
origindcmpath = r'C:\Users\fafafa\Desktop\wuda\result\8dcm'
originlist = glob.glob(os.path.join(origindcmpath,'*.dcm'))
originlist.sort()

preddcmpath = r'C:\Users\fafafa\Desktop\wuda\result\8512imagedcm'
predlist = glob.glob(os.path.join(preddcmpath,'*.dcm'))
predlist.sort()
savecontourpngpath = r'C:\Users\fafafa\Desktop\wuda\result\rg\\'
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
    if slice <= 9:  # 保存膨胀后的pre去原路经
        pred.save_as(savecontourpngpath + "0{}.dcm".format(slice))
    else:
        pred.save_as(savecontourpngpath + "{}.dcm".format(slice))






