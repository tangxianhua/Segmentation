import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
img = cv2.imread(r"H:\myvnet\Data\11.png")
def myrgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = g

    return gray
img = myrgb2gray(img)
# plt.imshow(img)
# plt.show()
arr = np.asarray(img)

print(arr[284][238])
# Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(Grayimg, 30, 255, cv2.THRESH_BINARY)
# cv2.imwrite( "testpng.png", thresh, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
# cv2.imshow('img',img)
# cv2.waitKey(0)


# dcmpath = r'H:\myvnet\codetest\00.dcm'
# ds = pydicom.read_file(dcmpath)
# print("开始读取DCM：", ds.PatientName)
# img = cv2.imread(r'H:\myvnet\codetest\testpng.png')

# print("开始读取PNG：", skeletonpath)

# matrix = np.asarray(img)
# arr = matrix.flatten()
# for z in range(262144):
#     if arr[z] < 100:
#         ds.pixel_array.flat[z] = 0
# ds.PixelData = ds.pixel_array.tostring()
# print("开始存入：", ds.PatientName)
# ds.save_as("test.dcm")

# print('开始存入',savecorppath+'.dcm')
# img = cv2.imread(r'H:\myvnet\Data\14.png')
# arr = np.asarray(img)
#
# for i in arr:
#     print(i)
# arr= np.asarray(img)
# A = np.max(arr)
# print(A)