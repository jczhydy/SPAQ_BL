

from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm
import os
from torch.utils.data import Dataset

#change them with your own file path
#traindata2.xlsx  stores the MOS of image for training,so as test2.xlsx 
file_name1 = '/root/IQA/traindata2.xlsx'
file_name2 = '/root/IQA/testdata2.xlsx'
data1 = pd.read_excel(file_name1,usecols=[0],header=None)
data1 = np.array(data1)
data1 = torch.Tensor(data1)
data2 = pd.read_excel(file_name2,usecols=[0],header=None)
data2 = np.array(data2)
data2 = torch.Tensor(data2)


def load_sample(sample_dir):
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
        lfilenames.sort(key=lambda x:int(x[-9:-4])) 
    if 'test' in sample_dir:
        for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        #遍历图片
            for filename in filenames:
            #每张图片的路径名
                filename_path = os.sep.join([dirpath,filename])
            #添加文件名
                lfilenames.append(filename_path)
            #添加文件名对应的标签
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
            img= Image.open(item[index],'r').convert("RGB")
            img = np.array(img)
            img = self.transform(img)
            return img,label
        if self.dataset_type=='test':
            img = Image.open(item[index],'r').convert("RGB")
            img = np.array(img)
            img = self.transform(img)
            return img,label
    def __len__(self):
        return len(self.sample_list)


train_transform=transforms.Compose([transforms.ToPILImage(),
                              transforms.CenterCrop(224),
                              #transforms.RandomHorizontalFlip(p=0.5),
                              transforms.ToTensor()
                              #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])
test_transform=transforms.Compose([transforms.ToPILImage(),
                              #transforms.RandomHorizontalFlip(p=0.5),
                              transforms.ToTensor()
                              #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])
loss_criterion = nn.L1Loss(reduction = 'sum') 
epochs=40

net=resnet50(pretrained=True)
channel_in = net.fc.in_features
net.fc=nn.Linear(channel_in,1)
net=net.cuda()

def train_val(net, data_loader, train_optimizer):
    total_loss, total_num, data_bar = 0.0, 0, tqdm(data_loader)
    for data, target in data_bar:
            len=data.size(0)
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)
            
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
                
            out_1,target_1 =out.cpu().detach().numpy(),target.cpu().detach().numpy()
            total_num += data.size(0)
            total_loss += loss.item()
            


            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} '
                                     .format('Train', epoch, epochs, total_loss / total_num
                                            ))

    return total_loss/total_num

def test_val(net, data_loader):
    test_loss, test_num, data_bar = 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data, target in data_bar:
            len=data.size(0)
            data, target = data.cuda(), target.cuda()

            # data1 = data[:,:,32:256,32:256]
            # data2 = data[:,:,256:480,32:256]
            # data3 = data[:,:,32:256,256:480]
            # data4 = data[:,:,256:480,256:480]
            data1 = data[:,:,:224,288:512]
            data2 = data[:,:,288:512,:224]
            data3 = data[:,:,:224,288:512]
            data4 = data[:,:,288:512,288:512]
            
            out1 = net(data1)
            out2 = net(data2)
            out3 = net(data3)
            out4 = net(data4)
            out=(out1+out2+out3+out4)/4
            
            loss=loss_criterion(out, target)
            
            test_num += len
            test_loss += loss.item()
            


            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} '
                                     .format('Test', epoch, epochs, test_loss / test_num
                                            ))

    return test_loss/test_num

if __name__ == '__main__':

    #change them with your own file path
    train_directory_rand='/root/IQA/rand_train_4'
    test_directory_rand='/root/IQA/rand_test_4'
    
    train_filenames_rand,train_labels_rand = load_sample(train_directory_rand)
    test_filenames_rand,test_labels_rand = load_sample(test_directory_rand)
    data_train = MyDataSet(dataset_type='train',filenames=train_filenames_rand,labelnames=train_labels_rand)
    data_test = MyDataSet(dataset_type='test',filenames=test_filenames_rand,labelnames=test_labels_rand)
    train_loader = Data.DataLoader(dataset=data_train,batch_size=8
                                   ,shuffle=True,num_workers=7)
    test_loader = Data.DataLoader(dataset=data_test,batch_size=16 ,shuffle=False,num_workers=7)


    for param in net.parameters():
         param.requires_grad = False
        
    results = {'train_loss': [],
               'test_loss':[]
             
             
               }

    for epoch in range(1, epochs + 1):
        if epoch<5:
            for param in net.fc.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(net.parameters(), lr=1e-3)

          
        if epoch>=5 and epoch<20:
            for param in net.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(net.parameters(), lr=1e-4)

            
        if epoch>=20 and epoch<30:

            optimizer = optim.Adam(net.parameters(), lr=1e-5)
 
            
        if epoch>=30 and epoch<40:
            optimizer = optim.Adam(net.parameters(), lr=1e-6)      

        net.train()
        train_loss=train_val(net, train_loader, optimizer)
        net.eval()
        test_loss = test_val(net,test_loader)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)      
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('result.csv', index_label='epoch')
        
    torch.save(net.state_dict(), 'image_model.pth')
