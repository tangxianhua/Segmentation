import glob
import os
import cv2
import numpy as np
import pydicom

# dcmpath = r'C:\Users\fafafa\Desktop\wuda\result\randomdcm'
# dcmlist = glob.glob(os.path.join(dcmpath,'*.dcm'))
# dcmlist.sort()
# pngpath = r'C:\Users\fafafa\Desktop\wuda\result\8512image'
# pnglist = glob.glob(os.path.join(pngpath,'*.png'))
# pnglist.sort()
# contourdcmpath = r'C:\Users\fafafa\Desktop\wuda\result\8512imagedcm\\'
# for slice in range(32):
#     if slice<=9:
#         name1 = '0'+str(slice)
#     else:
#         name1 = str(slice)
#     ds = pydicom.read_file(dcmlist[slice])
#     print("开始读取：", ds.PatientName)
#     img = cv2.imread(pnglist[slice])
#     Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(Grayimg, 30, 255, cv2.THRESH_BINARY)
#     matrix = np.asarray(thresh)
#     arr = matrix.flatten()
#     for i in range(262144):
#         if arr[i] < 100:
#             ds.pixel_array.flat[i] = 0
#     ds.PixelData = ds.pixel_array.tostring()
#     ds.save_as(contourdcmpath + name1 + ".dcm")


#rg2bindarydcm
dcmpath = r'C:\Users\fafafa\Desktop\wuda\result\rg'
dcmlist = glob.glob(os.path.join(dcmpath,'*.dcm'))
dcmlist.sort()

contourdcmpath = r'C:\Users\fafafa\Desktop\wuda\result\rgbindary\\'
for slice in range(32):
    if slice<=9:
        name1 = '0'+str(slice)
    else:
        name1 = str(slice)
    ds = pydicom.read_file(dcmlist[slice])

    for i in range(262144):
        if ds.pixel_array.flat[i] != 0:
            ds.pixel_array.flat[i] = 65535
    ds.PixelData = ds.pixel_array.tostring()
    ds.save_as(contourdcmpath + name1 + ".dcm")



