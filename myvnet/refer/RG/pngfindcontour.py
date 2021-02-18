import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
filepath = r'C:\Users\fafafa\Desktop\wuda\result\8512image'
savepath = r'C:\Users\fafafa\Desktop\wuda\result\8512contourimage\\'
pnglist = glob.glob(os.path.join(filepath,'*.png'))
pnglist.sort()
for i in range(32):
    pngimage = cv2.imread(pnglist[i])
    pngimage = pngimage[:,:,1]
    countourempty  = np.zeros(shape=(512,512))
    for a in range(512):
        for b in range(512):
            if pngimage[a][b] != 0:
                for c in range(-1, 2):
                    for d in range(-1, 2):
                        if (pngimage[a + c][b + d] == 0):
                            countourempty[a][b] = 255
    if i<=9:#保存边缘png
      cv2.imwrite(savepath +"0{}.png".format(i),countourempty)
    else:
      cv2.imwrite(savepath+"{}.png".format(i),countourempty)
