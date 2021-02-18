#分组label
import os,inspect
import sys
from PIL import Image
import  cv2
import numpy as np
#import matplotlib.pyplot as plt
import shutil



name1 = 'I:\changedata\\标准化train'
for i in range(4,51):
   if i<=9:
      name2 = '0' + str(i)
   else:
      name2 = str(i)
   filepath = os.path.join(name1,name2,'json')
   #filepath = r"I:\changedata\标准化train\01\json"
   j=0
   for root, dirs, files in os.walk(filepath):
      for file in files:
             if file == 'label.png':
                    filepath = os.path.join(root, file)
                    filename = os.path.basename(file)
                    basename = os.path.basename(file)
                    fragments = root.split('\\')
                    name = 'origin2part'
                    directpath = os.path.join(fragments[0], fragments[1], fragments[2],fragments[3])
                    directpath2=os.path.join(directpath, name)
                    print(directpath2)
                    #shutil.copy(filepath, directpath2)
                    oldname = os.path.join(filepath)
                    if ((j+32)%32) <=9:
                     newname = os.path.join(directpath2, '0'+str((j+32)%32) + '.png')
                    else:
                     newname = os.path.join(directpath2, str((j + 32) % 32) + '.png')
                    os.rename(oldname, newname)
                    print(oldname, '======>', newname)
                    j+=1


