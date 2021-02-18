#给每个病人建立label文件夹
import os,inspect
import sys
from PIL import Image
import  cv2
import numpy as np
#import matplotlib.pyplot as plt
import shutil

# filepath = r"I:\changedata\xmodule"
#
# for root, dirs, files in os.walk(filepath):
#     for file in files:
#         if file == 'label.png':
#             name = os.path.join(root, file)
#             fragments = root.split('\\')
#             mkpath = os.path.join(fragments[0], fragments[1], fragments[2])
#             name = './label'
#             mkpath2 = os.path.join(mkpath, name)
#             if not os.path.exists(mkpath2):
#                 os.mkdir(mkpath + './label')
# for i in range(1,51):
#     if i <= 9:
#       path = os.path.join(filepath,'0'+str(i),'C32')
#     else:
#       path = os.path.join(filepath, str(i), 'C32')
#     #print(path)
#     os.mkdir(path)

#提取所有车辆灰度图和原图
# labelfilepath = r"H:\cheshen_zuoqian\hao"
# for root, dirs, files in os.walk(labelfilepath):
#     for file in files:
#         rootname = 'H:\cheshen_zuoqian'
#         originame = 'hao_origin'
#         greyname = 'hao_grey'
#         targetname = '6'
#         if os.path.splitext(file)[1] == '.jpg':
#             name = os.path.splitext(file)[0]
#             print(name)
#             name1 = (name[:-6])
#             #print(name[:-6])
#             newpath1 = name1+'.jpg'#原图
#             #print(newpath1)
#             newpath2 = name1 + '_grey'+'.jpg'#灰度图
#             #print(newpath2)
#             repath1 = os.path.join(rootname,targetname,newpath1)#target里原图路径
#             print(repath1)
#             repath2 = os.path.join(rootname,targetname,newpath2)#target里灰度图路径
#             print(repath2)
#             finalpath1 = os.path.join(rootname,originame,newpath1)#移动后原图路径
#             #finalpath1 = os.path.join(rootname, originame)  # 移动后原图路径
#             print(finalpath1)
#             finalpath2 = os.path.join(rootname, greyname,newpath2)#移动后灰度图路径
#             print(finalpath2)
#             if os.path.exists(repath1):
#                os.rename(repath1, finalpath1)
#                #shutil.copy(repath1, 'finalpath1')
#             if os.path.exists(repath2):
#                os.rename(repath2, finalpath2)
#                #shutil.copy(repath2, 'finalpath2')


#遍历所有车辆图片写入文件
# labelfilepath = r"H:\cheshen_zuoqian\hao_grey"
# txtsavepath = 'H:\cheshen_zuoqian'
# txtname = 'hao_grey.txt'
# for root, dirs, files in os.walk(labelfilepath):
#     with open(os.path.join(txtsavepath, str(txtname)), 'w') as f:
#      for file in files:
#         if os.path.splitext(file)[1] == '.jpg':
#             filename = os.path.basename(file)
#             print(filename)
#             f.write(str(filename)+'\n')

#批量提取中线点
# path = r'I:\center'
# bpng = 'bpng'
# skeleton ='skeleton'
# for i in range(1,51):
#     if i <=9:
#         bianlipath= os.path.join(path,'0'+str(i))
#         savepath = os.path.join(path,'0'+str(i)+skeleton)
#         for root, dirs, files in os.walk(bianlipath):
#             for file in files:
#                 img = cv2.imread(file)
#                 array = np.array(img)
#                 skeleton = morphology.skeletonize(img)
#                 cv2.imwrite(savepath + '0'+str(i)+'.png', skeleton)
#
#     else:
#         bianlipath= os.path.join(path,str(i))
#         savepath = os.path.join(path,str(i)+skeleton)
#         for root, dirs, files in os.walk(bianlipath):
#             for file in files:
#                 img = cv2.imread(file)
#                 array = np.array(img)
#                 skeleton = morphology.skeletonize(img)
#                 cv2.imwrite(savepath + str(i)+'.png', skeleton)
#
#

import os
import glob
import cv2
import numpy as np

# from matplotlib import pylab as plb
for a in range(1,53):
    if a <=9:
        name = '0'+str(a)
    else:
        name = str(a)
    allpath = r'H:\neural computing\otherdataset\Pancreas-CT\coding\seleted\labelbinary'
    path = os.path.join(allpath,name)
    saveallpath = r'H:\neural computing\otherdataset\Pancreas-CT\coding\seleted\labelbinary2'
    savepath = os.path.join(saveallpath,name)
    print(path)
    print(savepath)
    filelist = glob.glob(os.path.join(path,'*.png'))
    filelist.sort(key=lambda x:int(x.split('\\')[-1][:-4]))
    #filelist.sort(key=lambda x:int(x[:-4][68:]))
    for j in range(200):
        img = cv2.imread(filelist[j])
        img = img[:, :, 0]
        matrix = np.asarray(img)
        newimg = np.zeros((310, 362))
        for c in range(310):
            for d in range(362):
                if ((c > 40) & (c < 260)) & ((d > 60) & (d < 280)):
                    newimg[c][d] = matrix[c][d]
                else:
                    newimg[c][d] = 0
        if j <=9:
          cv2.imwrite(savepath +'\\'+'0'+str(j) + ".png",newimg)

        else:
          cv2.imwrite(savepath + '\\'+str(j) + ".png",newimg)



