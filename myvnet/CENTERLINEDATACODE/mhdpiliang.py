# -*- coding: utf-8 -*-
import cv2
import os
import pydicom
import numpy
import SimpleITK

# 路径和列表声明

allpath = 'I:\\MASKcenter256'
subone = 'resize256'
#subtwo = 'resize256'
savepath = 'I:\\mhdsave'

for a in range(1, 51):
    print('循环')
    lstFilesDCM = []
    if a <= 9:
        path = os.path.join(allpath, '0'+str(a),subone)

    else:
        path = os.path.join(allpath, str(a),subone)

    for i in os.listdir(path):
      dcm_path = os.path.join(os.path.abspath(path), i)
      lstFilesDCM.append(dcm_path)

    lstFilesDCM.sort()
    for b in range(len(lstFilesDCM)):
       print(lstFilesDCM[b])

    # 第一步：将第一张图片作为参考图片，并认为所有图片具有相同维度
    RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

    # 第二步：得到dicom图片所组成3D图片的维度
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))  # ConstPixelDims是一个元组

    # 第三步：得到x方向和y方向的Spacing并得到z方向的层厚
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # 第四步：得到图像的原点
    Origin = RefDs.ImagePositionPatient

    # 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype
    ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)  # array is a numpy array

    # 第五步:遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    j = 0
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        # cv2.imwrite("out_" + str(i) + ".png", ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)])
        j += 1

    # 第六步：对numpy数组进行转置，即把坐标轴（x,y,z）变换为（z,y,x）,这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
    ArrayDicom = numpy.transpose(ArrayDicom, (2, 0, 1))
    # 第七步：将现在的numpy数组通过SimpleITK转化为mhd和raw文件
    sitk_img = SimpleITK.GetImageFromArray(ArrayDicom, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)
    if a <=9:
      SimpleITK.WriteImage(sitk_img, os.path.join(savepath, '0'+str(a)+"Sample_segmentation" + ".mhd"))
    else:
      SimpleITK.WriteImage(sitk_img, os.path.join(savepath,  str(a) + "Sample_segmentation" + ".mhd"))
    #_segmentation