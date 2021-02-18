# %% -*- coding: utf-8 -*-
'''
Author: Shreyas Padhy
Driver file for Standard UNet Implementation
'''
from __future__ import print_function
import argparse
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import torch.utils.data as Data
import torchvision.transforms as tr
import torch.nn.parallel
from loss import SoftDiceLoss
from vnet import VNet
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=True,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=256, metavar='N',
                    help='imsize')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--data-folder', type=str, default='./Data/', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--optimizer', type=str, default='ADAM', metavar='str',
                    help='Optimizer (default: SGD)')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
DATA_FOLDER = args.data_folder

weightsavepath = ''
txtsavepath = ''
txtname     = ''
pngsavepath = ''
pngname     = ' '

# ********************************训练数据*********************************************
print('********************************训练数据*********************************************')
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据1@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/trainone.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
torch_dataset1 = Data.TensorDataset(x, y)
trainloader1 = Data.DataLoader(
    dataset=torch_dataset1,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据2@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/traintwo.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
torch_dataset2 = Data.TensorDataset(x, y)
trainloader2 = Data.DataLoader(
    dataset=torch_dataset2,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据3@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/trainthree.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
torch_dataset3 = Data.TensorDataset(x, y)
trainloader3 = Data.DataLoader(
    dataset=torch_dataset3,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据4@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/trainfour.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
torch_dataset4 = Data.TensorDataset(x, y)
trainloader4 = Data.DataLoader(
    dataset=torch_dataset4,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@导入训练数据5@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/trainfive.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
torch_dataset5 = Data.TensorDataset(x, y)
trainloader5 = Data.DataLoader(
    dataset=torch_dataset5,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# ********************************验证数据*********************************************
print('********************************验证数据*********************************************')
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据1@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/validationone.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
# 把数据放在数据库中
vali_dataset1 = Data.TensorDataset(x, y)
validationloder1 = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=vali_dataset1,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据2@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/validationtwo.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
vali_dataset2 = Data.TensorDataset(x, y)
validationloder2 = Data.DataLoader(
    dataset=vali_dataset2,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据3@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/validationthree.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
vali_dataset3 = Data.TensorDataset(x, y)
validationloder3 = Data.DataLoader(
    dataset=vali_dataset3,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据4@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/validationfour.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
vali_dataset4 = Data.TensorDataset(x, y)
validationloder4 = Data.DataLoader(
    dataset=vali_dataset4,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@验证数据5@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
with open('/content/drive/My Drive/Data/P3/validationfive.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
vali_dataset5 = Data.TensorDataset(x, y)
validationloder5 = Data.DataLoader(
    dataset=vali_dataset5,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# ********************************测试数据*********************************************
print('********************************测试数据*********************************************')
with open('/content/drive/My Drive/Data/test_data.p3', 'rb') as f:
    x, y = pickle.load(f)
x = x.reshape(x.shape + (1,)).astype(np.float32)
y = y.reshape(y.shape + (1,)).astype(np.float32)  # (50, 256, 256, 32, 1)
x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.Tensor.permute(x, (0, 4, 3, 2, 1))
y = torch.Tensor.permute(y, (0, 4, 3, 2, 1))
print(x.shape)
print(y.shape)
test_dataset = Data.TensorDataset(x, y)
testloder = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)
# ******************************************************************************************
model = VNet()
if args.cuda:
    model.cuda()
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Defining Loss Function
criterion = SoftDiceLoss()
testcriterion = SoftDiceLoss()


def confusionmetric(X, Y):
    x1 = X.reshape(-1)  # pickle mask最大值为1，mhd为255
    # print('x1 is:',np.max(x1))
    y1 = Y.reshape(-1)
    # print('y1 is:', np.max(y1))
    conmat = confusion_matrix(x1, y1)
    com = conmat.flatten()
    TN = com[0]
    FP = com[1]
    FN = com[2]
    TP = com[3]
    return TN, FP, FN, TP


# **************************************训练****************************************
# *******************************************************************************************
# ****************************** train1 ***************************************************

def train(epoch):
    print('******************************train epoch******************', epoch)
    #t1,v1
    train1loss = 0
    t1acc = 0
    t1sens = 0
    t1spec = 0
    t1ppv = 0
    t1npv = 0
    vali1loss = 0
    v1acc = 0
    v1sens = 0
    v1spec = 0
    v1ppv = 0
    v1npv = 0

    #t2,v2
    train2loss = 0
    t2acc = 0
    t2sens = 0
    t2spec = 0
    t2ppv = 0
    t2npv = 0
    vali2loss = 0
    v2acc = 0
    v2sens = 0
    v2spec = 0
    v2ppv = 0
    v2npv = 0

    #t3,v3
    train3loss = 0
    t3acc = 0
    t3sens = 0
    t3spec = 0
    t3ppv = 0
    t3npv = 0
    vali3loss = 0
    v3acc = 0
    v3sens = 0
    v3spec = 0
    v3ppv = 0
    v3npv = 0

    #t4,v4
    train4loss = 0
    t4acc = 0
    t4sens = 0
    t4spec = 0
    t4ppv = 0
    t4npv = 0
    vali4loss = 0
    v4acc = 0
    v4sens = 0
    v4spec = 0
    v4ppv = 0
    v4npv = 0

    #t5,v5
    train5loss = 0
    t5acc = 0
    t5sens = 0
    t5spec = 0
    t5ppv = 0
    t5npv = 0
    vali5loss = 0
    v5acc = 0
    v5sens = 0
    v5spec = 0
    v5ppv = 0
    v5npv = 0

    model.train()
#**************************************************第一次循环
    print('**********************************train1***********************')
    for batch_idx, (image, mask) in enumerate(trainloader1):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('Train1 loss is :', loss.item())
        loss.backward()
        optimizer.step()
        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)

        train1loss += loss.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        t1acc += ACC
        SENS = TP / (TP + FN)
        t1sens += SENS
        SPEC = TN / (TN + FP)
        t1spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        t1ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        t1npv += NPV

        model_path = os.path.join(weightsavepath,'ceTrainweight--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    train1loss = train1loss / 40
    t1acc = t1acc / 40
    t1sens = t1sens / 40
    t1spec = t1spec / 40
    t1ppv = t1ppv / 40
    t1npv = t1npv / 40

    print('******************************validation1*******************************')
    vloader1 = validationloder1
    for batch_idx, (image, mask) in tqdm(enumerate(vloader1)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        vali1loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        v1acc += ACC
        SENS = TP / (TP + FN)
        v1sens += SENS
        SPEC = TN / (TN + FP)
        v1spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        v1ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        v1npv += NPV
    vali1loss += vali1loss / 10
    v1acc += v1acc / 10
    v1sens += v1sens / 10
    v1spec += v1spec / 10
    v1ppv += v1ppv / 10
    v1npv += v1npv / 10
#*************************************************第二次循环
    for batch_idx, (image, mask) in enumerate(trainloader2):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('Train2 loss is :', loss.item())
        loss.backward()
        optimizer.step()
        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)

        train2loss += loss.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        t2acc += ACC
        SENS = TP / (TP + FN)
        t2sens += SENS
        SPEC = TN / (TN + FP)
        t2spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        t2ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        t2npv += NPV

        model_path = os.path.join(weightsavepath,'ceTrainweight--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    train2loss = train2loss / 40
    t2acc = t2acc / 40
    t2sens = t2sens / 40
    t2spec = t2spec / 40
    t2ppv = t2ppv / 40
    t2npv = t2npv / 40

    print('******************************validation2*******************************')
    vloader2 = validationloder2
    for batch_idx, (image, mask) in tqdm(enumerate(vloader2)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        vali2loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        v2acc += ACC
        SENS = TP / (TP + FN)
        v2sens += SENS
        SPEC = TN / (TN + FP)
        v2spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        v2ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        v2npv += NPV
    vali2loss += vali2loss / 10
    v2acc += v2acc / 10
    v2sens += v2sens / 10
    v2spec += v2spec / 10
    v2ppv += v2ppv / 10
    v2npv += v2npv / 10
# *************************************************第3次循环
    for batch_idx, (image, mask) in enumerate(trainloader3):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('Train3 loss is :', loss.item())
        loss.backward()
        optimizer.step()
        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)

        train3loss += loss.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        t3acc += ACC
        SENS = TP / (TP + FN)
        t3sens += SENS
        SPEC = TN / (TN + FP)
        t3spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        t3ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        t3npv += NPV

        model_path = os.path.join(weightsavepath,'ceTrainweight--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    train3loss = train3loss / 40
    t3acc = t3acc / 40
    t3sens = t3sens / 40
    t3spec = t3spec / 40
    t3ppv = t3ppv / 40
    t3npv = t3npv / 40

    print('******************************validation3*******************************')
    vloader3 = validationloder3
    for batch_idx, (image, mask) in tqdm(enumerate(vloader3)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        vali3loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        v3acc += ACC
        SENS = TP / (TP + FN)
        v3sens += SENS
        SPEC = TN / (TN + FP)
        v3spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        v3ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        v3npv += NPV
    vali3loss += vali3loss / 10
    v3acc += v3acc / 10
    v3sens += v3sens / 10
    v3spec += v3spec / 10
    v3ppv += v3ppv / 10
    v3npv += v3npv / 10
# *************************************************第4次循环
    for batch_idx, (image, mask) in enumerate(trainloader4):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('Train4 loss is :', loss.item())
        loss.backward()
        optimizer.step()
        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)

        train4loss += loss.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        t4acc += ACC
        SENS = TP / (TP + FN)
        t4sens += SENS
        SPEC = TN / (TN + FP)
        t4spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        t4ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        t4npv += NPV

        model_path = os.path.join(weightsavepath,'ceTrainweight--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    train4loss = train4loss / 40
    t4acc = t4acc / 40
    t4sens = t4sens / 40
    t4spec = t4spec / 40
    t4ppv = t4ppv / 40
    t4npv = t4npv / 40

    print('******************************validation4*******************************')
    vloader4 = validationloder4
    for batch_idx, (image, mask) in tqdm(enumerate(vloader4)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        vali4loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        v4acc += ACC
        SENS = TP / (TP + FN)
        v4sens += SENS
        SPEC = TN / (TN + FP)
        v4spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        v4ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        v4npv += NPV
    vali4loss += vali4loss / 10
    v4acc += v4acc / 10
    v4sens += v4sens / 10
    v4spec += v4spec / 10
    v4ppv += v4ppv / 10
    v4npv += v4npv / 10
# *************************************************第5次循环
    for batch_idx, (image, mask) in enumerate(trainloader5):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        print('Train5 loss is :', loss.item())
        loss.backward()
        optimizer.step()
        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)

        train5loss += loss.item()
        ACC = (TP + TN) / (TP + FN + FP + TN)
        t5acc += ACC
        SENS = TP / (TP + FN)
        t5sens += SENS
        SPEC = TN / (TN + FP)
        t5spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        t5ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        t5npv += NPV

        model_path = os.path.join(weightsavepath,'ceTrainweight--{}-{}'.format(args.epochs, args.lr))
        torch.save(model.state_dict(), model_path)
    train5loss = train5loss / 40
    t5acc = t5acc / 40
    t5sens = t5sens / 40
    t5spec = t5spec / 40
    t5ppv = t5ppv / 40
    t5npv = t5npv / 40

    print('******************************validation5*******************************')
    vloader5 = validationloder5
    for batch_idx, (image, mask) in tqdm(enumerate(vloader5)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        vali5loss += testcriterion(output, mask).item()

        output = output.data.byte().cpu().numpy()
        mask = mask.data.byte().cpu().numpy()
        TP, FN, FP, TN = confusionmetric(mask, output)
        # print(TN,FP,FN,TP)
        ACC = (TP + TN) / (TP + FN + FP + TN)
        v5acc += ACC
        SENS = TP / (TP + FN)
        v5sens += SENS
        SPEC = TN / (TN + FP)
        v5spec += SPEC
        PPV = TP / (TP + FP + 1e-10)
        v5ppv += PPV
        NPV = TN / (TN + FN + 1e-10)
        v5npv += NPV
    vali5loss += vali5loss / 10
    v5acc += v5acc / 10
    v5sens += v5sens / 10
    v5spec += v5spec / 10
    v5ppv += v5ppv / 10
    v5npv += v5npv / 10
#*********************计算五次平均***********************************
    TlossALL = (train1loss + train2loss + train3loss + train4loss + train5loss)/5
    TlossALL_list.append(TlossALL)
    Tdice = 1 - TlossALL
    Tdice_list.append(Tdice)
    Tacc = (t1acc + t2acc + t3acc + t4acc + t5acc) / 5
    Tacc_list.append(Tacc)
    Tsens = (t1sens + t2sens + t3sens + t4sens + t5sens) / 5
    Tsens_list.append(Tsens)
    Tspec = (t1spec + t2spec + t3spec + t4spec +t5spec) / 5
    Tspec_list.append(Tspec)
    Tppv = (t1ppv + t2ppv + t3ppv + t4ppv + t5ppv) / 5
    Tppv_list.append(Tppv)
    Tnpv = (t1npv + t2npv + t3npv + t4npv +t5npv) / 5
    Tnpv_list.append(Tnpv)

    VlossALL = (vali1loss + vali2loss + vali3loss + vali4loss + vali5loss)/5
    VlossALL_list.append(VlossALL)
    if (epoch >=80) & (VlossALL > VlossALL_list[epoch-1]):
      print('过拟合')
      print('best epoch is ',epoch-1)
      print('best Vloss is ', VlossALL_list[epoch-1])
    else:
      print('没有过拟合')
      print('Total Vloss is ', VlossALL)
      model_path = os.path.join(weightsavepath, 'cebestweight--{}-{}'.format(args.epochs, args.lr))
      torch.save(model.state_dict(), model_path)

    print_format = [epoch, TlossALL, Tdice, Tacc, Tsens, Tspec, Tppv, Tnpv]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(
        '===> Training step {} \tLoss: {:.7f}\tDice: {:.7f}\tAcc: {:.7f}\tSe: {:.7f}\tSp: {:.7f}\tPPV:{:.7f}\tNPV::{:.7f}'.format(
            *print_format))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with open(os.path.join(txtsavepath,str(txtname)), 'w') as f:
        f.write(str(TlossALL_list))
        f.write(str(Tdice_list))
        f.write(str(Tacc_list))
        f.write(str(Tsens_list))
        f.write(str(Tspec_list))
        f.write(str(Tppv_list))
        f.write(str(Tnpv_list))
    plt.plot(TlossALL_list)
    plt.savefig(os.path.join(pngsavepath,str(pngname)))


# **********************************************测试*********************************************8
def test(save_output=True):
    test_loss = 0
    loader = testloder
    # path = r'/content/drive/My Drive/Project/myvnet/Results/cebestweight'
    # print('load best weight')
    # model.load_state_dict(torch.load(path))

    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        image, mask = Variable(image), Variable(mask)
        output = model(image)
        test_loss += testcriterion(output, mask).item()
        print('test loss is ', test_loss)
        if save_output:
            np.save(r'/content/gdrive/My Drive/Project/myvnet/Results/{}-Pickle0204.npy'.format(batch_idx),
                    output.data.byte().cpu().numpy())


# **********************************************************************************************************
if args.train:
    # path = r'/content/gdrive/My Drive/Project/myvnet/Results/Pickle0204-1-300-0.0001'
    # model.load_state_dict(torch.load(path))
    # print('load again')

    VlossALL_list = []
    TlossALL_list = []
    Tdice_list = []
    Tacc_list = []
    Tsens_list = []
    Tspec_list = []
    Tppv_list = []
    Tnpv_list = []
    for epoch in range(args.epochs):
        train(epoch)

    test(save_output=True)

