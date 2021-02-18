import os
import shutil

filepath = r"C:\Users\fafafa\Desktop\9.7data\754122chengling\dcm"
n=0
for root, dirs, files in os.walk(filepath):
    files.sort(key=lambda x: int(x[:-4][1:]))
    for file in files:
        suf = os.path.splitext(file)[1]
        filename = os.path.basename(file)
        if suf == '.dcm':
            print(file)
            dst_file_path = filepath
            oldname = os.path.join(dst_file_path, filename)  # os.sep添加系统分隔符
            newname = os.path.join(dst_file_path, str(n) + '.dcm')
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
            n += 1
            #
