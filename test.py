
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet152
from torchvision.models.resnet import resnet50
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm
import os
from torch.utils.data import Dataset


#change it with your own file path
file_name2 = '/root/IQA/testdata2.xlsx'

data2 = pd.read_excel(file_name2,usecols=[0],header=None)
data2 = np.array(data2)
data2 = torch.Tensor(data2)

test_transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

def load_sample(sample_dir):    #
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
            #添加文件名对应的标签,即图片的MOS值
                labelnames.append(data1[t])
                t=t+1
    if 'test' in sample_dir:
        for (dirpath,dirnames,filenames) in os.walk(sample_dir):
            for filename in filenames:
            #每张图片的路径名
                filename_path = os.sep.join([dirpath,filename])
            #添加文件名
                lfilenames.append(filename_path)
            #添加文件名对应的标签,即图片的MOS值
                labelnames.append(data2[t])
                t=t+1 
    lfilenames.sort(key=lambda x:int(x[-9:-4]))
    lab = list(labelnames)   

    return lfilenames,lab

class MyDataSet(Dataset):
    def __init__(self,dataset_type,filenames, labelnames):
        if dataset_type=='train':
           self.transform = train_transform

        if dataset_type=='test':
           self.transform = test_transform
        self.dataset_type = dataset_type
        self.sample_list = filenames
        self.label_list = labelnames
        
    def __getitem__(self,index):
        item = self.sample_list
        label = self.label_list[index]
        label = torch.Tensor(label)
        if self.dataset_type=='train':
        # img = cv2.imread(item.split(' _')[0])
            im = Image.open(item[index],'r').convert("RGB")
            img = np.array(im)
            img = self.transform(img)
            return img,label
        
        if self.dataset_type=='test':
            img = Image.open(item[index],'r').convert("RGB")
            img = np.array(img)
            img = self.transform(img)
            img1 = img[:,:224,:224]
            img2 = img[:,288:512,:224]
            img3 = img[:,:224,288:512]
            img4 = img[:,288:512,288:512]

            return img1,img2,img3,img4,label       
        
 
    def __len__(self):
        return len(self.sample_list)

test_directory_rand='/root/IQA/rand_test_4'
test_filenames_rand,test_labels_rand = load_sample(test_directory_rand)
data_test = MyDataSet(dataset_type='test',filenames=test_filenames_rand,labelnames=test_labels_rand)

net=resnet50(pretrained=False)
channel_in = net.fc.in_features
net.fc=nn.Linear(channel_in,1)
net=net.cuda()
model_path='image_model24.pth'  #change it with your own file
state_dict = torch.load(model_path, map_location='cpu')
net.load_state_dict(state_dict, strict=True)

sum=0
pre=np.zeros(2000)  #change '2000' with the amount of images in your testdataset
lab=np.zeros(2000)
net.eval()
for i in range(2000):
    with torch.no_grad():
        data1,data2,data3,data4,label=data_test.__getitem__(i)    
        data1=data1.cuda()
        data2=data2.cuda()
        data3=data3.cuda()
        data4=data4.cuda()
        
        data1=torch.unsqueeze(data1,0)
        data2=torch.unsqueeze(data2,0)
        data3=torch.unsqueeze(data3,0)
        data4=torch.unsqueeze(data4,0)
         
        out1=net(data1)
        out2=net(data2)
        out3=net(data3)
        out4=net(data4)
        
        label=label.item()
        out=(out1+out2+out3+out4)/4
        pre[i]=out
        lab[i]=label
# GET RMSE
# ab=np.abs(pre-lab)  
# for i in ab:
#     sum=sum+i**2
# RMSE=np.sqrt(sum/2000)
# print(RMSE)
A= pd.Series(pre.tolist())
B= pd.Series(lab.tolist())

srocc=A.corr(B,method='spearman')
print(srocc)
