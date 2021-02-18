import os
import glob

path = r'I:\changedata\standtrain'
sub1 = '256'

obmake = 'chamfer'
for i in range(1,51):
    if i <=9:
        bianlipath= os.path.join(path,'0'+str(i),sub1)#01文件夹
        print(bianlipath)
        makename= os.path.join(bianlipath,obmake)
        #makename = os.path.join(path,'0'+str(i))
        print(makename)
        os.mkdir(makename)

    else:
        bianlipath= os.path.join(path,str(i),sub1)#01文件夹
        print(bianlipath)
        makename= os.path.join(bianlipath,obmake)
        #makename = os.path.join(path,str(i))
        print(makename)
        os.mkdir(makename)

#名字00排序
#
# path = r'I:\maskcenterline\27\maskcenterline\dcm'
#
# for i in range(0, 32):
#     if i <= 9:
#         #bianlipath = os.path.join(path, '0' + str(i))  # 01/maskcenterline文件夹
#
#         oldname = os.path.join(path,str(i)+'.dcm')
#         newname = os.path.join(path, '0' +str(i)+'.dcm')
#         os.rename(oldname, newname)
#         print(oldname, '======>', newname)
#
#     else:
#         oldname  =  os.path.join(path,str(i)+'.dcm')
#         print(oldname)


#加dcm后缀
# path = r'I:\changedata\valadation\10CaiJuGen\1S214020\S20'
# # paths = glob.glob(path)
# # paths.sort()
# for root,dir,files in os.walk(path):
#    for file in files:
#        name = os.path.splitext(file)[0]
#        suffix = os.path.splitext(file)[1]
#        if suffix == '.dcm':
#            print(file)
#        else:
#            oldname = os.path.join(path,file)
#            newname = os.path.join(path, file+'.dcm')
#            os.rename(oldname, newname)
#            print(oldname, '======>', newname)

#排序重命名
# filepath = r"I:\changedata\标准化train\32\json"
# n=1
# for root, dirs, files in os.walk(filepath):
#     files.sort(key=lambda x: int(x[:-5][11:]))
#     for file in files:
#         print(file)
#         filename = os.path.basename(file)
#         #print(filename)
#         dst_file_path = filepath
#         oldname = os.path.join(dst_file_path, filename)  # os.sep添加系统分隔符
#         if n <= 9:
#           newname = os.path.join(dst_file_path, '0'+str(n) + '.json')
#         else:
#           newname = os.path.join(dst_file_path,  str(n) + '.json')
#         os.rename(oldname, newname)
#         print(oldname, '======>', newname)
#         n += 1

