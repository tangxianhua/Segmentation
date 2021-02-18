import os
import shutil

filepath = r"G:\data\10152\189896luchangbao\dcm"

n = 0
for root, dirs, files in os.walk(filepath):
    for file in files:
        suf = os.path.splitext(file)[1]
        filename=os.path.basename(file)
        if suf== '.dcm':
            print(file)
            dst_file_path =filepath
            oldname = os.path.join(dst_file_path, filename)  # os.sep添加系统分隔符
            newname = os.path.join(dst_file_path, str(n) + '.dcm')
            # dst = r'D:\汤先华\data\bdclstm\sortdirectory'
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
            n += 1
            #
