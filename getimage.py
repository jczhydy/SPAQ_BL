import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

file_name1 = r'D:/SPAQ zip/traindata2.xlsx'
file_name2 = r'D:/SPAQ zip/testdata2.xlsx'
data1 = pd.read_excel(file_name1,usecols=[0],header=None)
data1 = np.array(data1)
data1 = torch.Tensor(data1)
data2 = pd.read_excel(file_name2,usecols=[0],header=None)
data2 = np.array(data2)
data2 = torch.Tensor(data2)

train_transform=transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(224),transforms.ToTensor()])
test_transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

def load_sample(sample_dir):
    #图片名列表
    t=0
    lfilenames = []
    #标签名列表
    labelnames = []
    if 'train' in sample_dir:
      for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        #遍历图片
        for filename in filenames:
            #每张图片的路径名
            filename_path = os.sep.join([dirpath,filename])
            #添加文件名
            lfilenames.append(filename_path)
            #添加文件名对应的标签
            labelnames.append(data1[t])
            t=t+1
    if 'test' in sample_dir:
      for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        for filename in filenames:
            #每张图片的路径名
            filename_path = os.sep.join([dirpath,filename])
            #添加文件名
            lfilenames.append(filename_path)
            #添加文件名对应的标签
            labelnames.append(data2[t])
            t=t+1 

    lab = list(labelnames)  

    return lfilenames,lab


train_directory = r'D:\SPAQ zip\train2'
test_directory = r'D:\SPAQ zip\test2'
train_filenames,train_labels = load_sample(train_directory)
test_filenames,test_labels = load_sample(test_directory)

def crop_train():
    t=1
    for i in train_filenames:
        img = Image.open(i)
        RandomCrop = transforms.CenterCrop(size=(512, 512))  #随机剪裁
        random_image = RandomCrop(img)
       # _resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
       #image=_resize(random_image)
        save_name = r'D:/SPAQ zip/rand_train_5/'+i[-9:-4]+'.png'
        random_image.save(save_name)
        t=t+1
def crop_test():
    t=1
    for i in test_filenames:
        img = Image.open(i)
        RandomCrop = transforms.CenterCrop(size=(512, 512))  
        random_image = RandomCrop(img)
        # image_resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        # image=image_resize(random_image)
        save_name = r'D:/SPAQ zip/rand_test_5/'+i[-9:-4]+'.png'
        random_image.save(save_name)
        t=t+1
crop_train()
crop_test()

