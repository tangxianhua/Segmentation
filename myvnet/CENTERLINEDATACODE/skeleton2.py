import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import os
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology,feature
import cv2
import pydicom
# Wpath = r'I:\maskcenterline'
# bpngpath = 'bpng'
# skeleton2 = 'skeleton2'
# for j in range(1,51):
#     if j <= 9:
#        filenum = '0'+str(j)
#     else:
#        filenum = str(j)
#     filepath = os.path.join(Wpath, filenum, bpngpath)
#     savepath = os.path.join(Wpath, filenum, skeleton2)
#     for i in range(0,32):
#         if i <= 9:
#             name = '0'+str(i)
#         else:
#             name = str(i)
#
#         # cesi= os.path.join(filepath, name + '.png')
#         # print(cesi)
#         # cesi2 = os.path.join(savepath, name + '.png')
#         # print(cesi2)
#
#         image = Image.open(os.path.join(filepath, name + '.png'))
#         skeleton = np.asarray(image)
#         skeleton1 = morphology.skeletonize_3d(skeleton)
#         cv2.imwrite(os.path.join(savepath,name+'.png'),skeleton1)
#
#         ##skeleton = skeleton/255
#         ## print(skeleton.shape)
#         ##skeleton1= skeleton1*255

# Wpath = r'I:\changedata\Test'
# maskpath1 = '256'
# maskpath2 = 'mask'
# skeletondcm = 'skeletondcm\\'
# for j in range(1,10):
#     if j <= 9:
#        filenum = '0'+str(j)
#     else:
#        filenum = str(j)
#     filepath = os.path.join(Wpath, filenum, maskpath1,maskpath2)
#     print(filepath)
#     savepath = os.path.join(Wpath, filenum, maskpath1,skeletondcm)
#     for root, dirs, files in os.walk(filepath):
#         files.sort(key=lambda x: int(x[:-5][5:]))
#         lala= 0
#         for file in files:
#             #print(os.path.join(filepath,file))
#
#
#         # cesi2 = os.path.join(savepath, name + '.png')
#         # print(cesi2)
#
#             image = pydicom.read_file(os.path.join(filepath,file))
#             print(os.path.join(filepath,file))
#             matrix = image.pixel_array
#             skeleton = morphology.skeletonize_3d(matrix)
#             skeleton1 = skeleton.flatten()
#             for t in range(65536):
#                 if skeleton1[t] == 0:
#                     image.pixel_array.flat[t] = 0
#             image.PixelData = image.pixel_array.tostring()
#             if lala <= 9:
#               image.save_as(savepath + '0'+str(lala) + ".dcm")
#               print(savepath + '0'+str(lala) + ".dcm")
#             else:
#               image.save_as(savepath + str(lala) + ".dcm")
#               print(savepath + str(lala) + ".dcm")
#             lala += 1


# Wpath = r'I:\changedata\Test'
# maskpath1 = '256'
# maskpath2 = 'mask'
# contourdcm = 'contourdcm\\'
# for j in range(1,10):
#     if j <= 9:
#        filenum = '0'+str(j)
#     else:
#        filenum = str(j)
#     filepath = os.path.join(Wpath, filenum, maskpath1,maskpath2)
#     print(filepath)
#     savepath = os.path.join(Wpath, filenum, maskpath1,contourdcm)
#     for root, dirs, files in os.walk(filepath):
#         files.sort(key=lambda x: int(x[:-5][5:]))
#         lala= 0
#         for file in files:
#
#             ds = pydicom.read_file(os.path.join(filepath,file))
#             contourempty = np.zeros(shape=(256, 256))
#             for i in range(256):
#                 for j in range(256):
#                     if ds.pixel_array[i][j] != 0:
#                         for x in range(-1, 2):
#                             for y in range(-1, 2):
#                                 if (ds.pixel_array[i + x][j + y] == 0):
#                                     contourempty[i][j] = 1
#             matrix = np.asarray(contourempty)
#             matrix1 = matrix.flatten()
#             for t in range(65536):
#                 if matrix1[t] == 0:
#                     ds.pixel_array.flat[t] = 0
#             ds.PixelData = ds.pixel_array.tostring()
#             if lala <= 9:
#               ds.save_as(savepath + '0'+str(lala) + ".dcm")
#               print(savepath + '0'+str(lala) + ".dcm")
#             else:
#               ds.save_as(savepath + str(lala) + ".dcm")
#               print(savepath + str(lala) + ".dcm")
#             lala += 1

#chamfermask
from scipy import ndimage
Wpath = r'I:\changedata\Test'
maskpath1 = '256'
maskpath2 = 'mask'
skeletondcm = 'chamfer\\'
for j in range(1,51):
    if j <= 9:
       filenum = '0'+str(j)
    else:
       filenum = str(j)
    filepath = os.path.join(Wpath, filenum, maskpath1,maskpath2)
    print(filepath)
    savepath = os.path.join(Wpath, filenum, maskpath1,skeletondcm)
    for root, dirs, files in os.walk(filepath):
        files.sort(key=lambda x: int(x[:-5][5:]))
        lala= 0
        for file in files:
            image = pydicom.read_file(os.path.join(filepath, file))
            for t in range(65536):
                if image.pixel_array.flat[t] > 0:
                    image.pixel_array.flat[t] = 1
            image.PixelData = image.pixel_array.tostring()
            matrix2 = image.pixel_array
            A = ndimage.distance_transform_edt(matrix2)
            A = np.asarray(A)
            for c in range(256):
                for d in range(256):
                    A[c][d] = round(A[c][d])
            M = np.max(A)
            for a in range(256):
                for b in range(256):
                    if A[a][b] != 0:
                        A[a][b] = M - A[a][b]
            A = A.flatten()
            for z in range(65536):
                image.pixel_array.flat[z] = A[z]
            image.PixelData = image.pixel_array.tostring()

            if lala <= 9:
              image.save_as(savepath + '0'+str(lala) + ".dcm")
              print(savepath + '0'+str(lala) + ".dcm")
            else:
              image.save_as(savepath + str(lala) + ".dcm")
              print(savepath + str(lala) + ".dcm")
            lala += 1