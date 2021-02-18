import os
import shutil

filepath = r"C:\Users\fafafa\Desktop\9.7data\754122chengling\dcm"


for root, dirs, files in os.walk(filepath):
    for file in files:
        filename = os.path.basename(file)
        filename_fragments= filename.split('.')
        dispath = filepath

        oldname = os.path.join(dispath,filename)
        newname = os.path.join(dispath,filename_fragments[9]+'.dcm')
        os.rename(oldname,newname)
        print(oldname,'====>',newname)