
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import os


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('cuda_dev', type=int,
                    help='CUDA Device ID')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_dev)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


class CurrencyDataset(Dataset):
    def __init__(self,root_dir):
        self.dir = os.path.abspath(root_dir)
    def __getitem__(self,idx):
        data_path = os.path.join(self.dir,'Batches','Batch_' + str(idx) + '.npy')
        label_path = os.path.join(self.dir,'Labels','Label_' + str(idx) + '.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return (data,labels)
    def __len__(self):
        return len(os.listdir(os.path.join(self.dir,'Batches')))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,512,(1,1))
        self.conv2 = nn.Conv2d(512,512,(3,1),padding=(1,0))
        self.conv3 = nn.Conv2d(512,512,(3,1),padding=(1,0))
        self.conv4 = nn.Conv2d(512,1024,(3,1))
        self.conv5 = nn.Conv2d(1024,1024,(3,1))
        self.conv6 = nn.Conv2d(1024,1024,(3,1))
        self.conv7 = nn.Conv2d(1024,2048,(3,1))
        self.conv8 = nn.Conv2d(2048,2048,(3,1))
        self.conv11 = nn.Conv2d(2048,2048,(4,1))
        self.conv9 = nn.Conv2d(2048,2048,(3,1))
        self.conv10 = nn.Conv2d(2048,1,(1,1))
        self.activation = F.leaky_relu
        self.pool = nn.AvgPool2d((3,1))
        self.norm_512 = nn.BatchNorm2d(512)
        self.norm_1024 = nn.BatchNorm2d(1024)
        self.norm_2048 = nn.BatchNorm2d(2048)
        self.norm_4096 = nn.BatchNorm2d(4096)
        self.norm_10000 = nn.BatchNorm2d(10000)
#         self.conv_test = nn.Conv2d(5,128,(48,3))
#         self.conv_test2 = nn.Conv2d(128,5,(3,1))
        self.fc1 = nn.Linear(5, 5)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(3000, 3500)
        self.fc3 = nn.Linear(3500, 5)

    def forward(self, x):
     #   x = self.dp1(x)
        
        x = self.activation(self.norm_512(self.conv1(x)))
        res = x
        x = self.activation(self.norm_512(self.conv2(x)))
        x += res
        x = self.activation(self.pool(x))
#         print(x.shape)
        x = self.activation(self.norm_512(self.conv3(x)))
#         print(x.shape)
        x = self.activation(self.norm_1024(self.conv4(x)))
#         print(x.shape)
        x = self.activation(self.norm_1024(self.conv5(x)))
#         print(x.shape)
        x = self.activation(self.norm_1024(self.conv6(x)))
#         print(x.shape)
        x = self.activation(self.norm_2048(self.conv7(x)))
#         print(x.shape)
        x = self.activation(self.norm_2048(self.conv8(x)))
        x = self.activation(self.norm_2048(self.conv9(x)))
        x = self.activation(self.norm_2048(self.conv11(x)))
        x = self.conv10(x)
#         print(x.shape)
#         x = F.relu(self.conv_test(x))
#         x = self.conv_test2(x)
        
        print(x.shape)
#         print(x.shape)
#         x = x.view(-1,5)
#         print(x.shape)
#         x = x.view(-1,5)
#         print(x.shape)       
#         x = F.relu(self.fc1(x))
#         print(x.shape)
# #         x = self.dp1(x)
#         x = F.relu(self.fc2(x))
# #         x = self.dp1(x)
#         x = self.fc3(x)
        x = x.view(-1,5)
#         x = self.dp1(x)
#         print(x.shape)
        
    
        x = F.softmax(x,dim=1)
        return x
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


# In[4]:


torch.backends.cudnn.deterministic = False
torch.manual_seed(991)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
data = CurrencyDataset('./Processed')
device = 'cuda:0'
currency_count = 5
net = Net()
print(net)
# net = nn.DataParallel(net,device_ids=[0])
net.cuda()
# net = net.half()
# for layer in net.modules():
#     if isinstance(layer, nn.BatchNorm2d):
#         layer.float()
# net = net.to(device)
net.train()
#dataloader = DataLoader(data,batch_size=1024,pin_memory=True, sampler=SubsetRandomSampler(list(range(1024,2048))))
dataloader = DataLoader(data,batch_size=512,pin_memory=True,shuffle=True)


# In[5]:


optimizer = optim.Adam(net.parameters(), lr=0.00015)


# In[6]:


net.apply(weight_reset)
# test_X, test_y = iter(dataloader).next()
# print(test_X)
# print(test_y)
print_every = 16
begin = time.time()
for epoch in range(100):
    epoch_loss = 0.0
    running_loss = 0.0
    j = 0
    for i,batch in enumerate(dataloader, 0):
        j = i
        optimizer.zero_grad()
        d,y = batch
        d = d.float()
        y = y.float()
        saved = y
        y[y <= 0.0] = 1.0
        y[y > 3.0] = 1.0
        y = torch.tan(-y - 1)
        #y = y ** -1
        #y = (y ** -1) ** 2
        if np.isnan(saved).any():
            print('Error')
            continue
#         d = test_X
#         y = test_y
        d = d.cuda()
        x = net(d)
        y = (y.reshape([-1,5])).cuda()
#         print(x.shape)
#         print(y.shape)
        readable_loss = torch.mul(x,saved.reshape([-1,5]).cuda()).sum(dim=1)
        s = readable_loss.shape[0]
        readable_loss = readable_loss.sum()/s
        z = torch.mul(x,y)
        z = z.sum(dim=1)
        loss = z.sum()
        loss /= z.shape[0]
#         loss = loss ** -1
        epoch_loss += readable_loss.item() * 10000
        running_loss += readable_loss.item() * 10000
        
        if i % print_every == print_every - 1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_every))
            print('Time elapsed: {} minutes'.format((time.time() - begin)/60))
            running_loss = 0.0

        loss.backward()
        optimizer.step()
    print('Epoch {} loss: {}'.format(epoch+1, epoch_loss/(j+1)))
end = time.time()
print("FINAL TIME: {}".format(end-begin))


# In[ ]:


# net.eval()
# for i,batch in enumerate(dataloader):
#     d,y = batch
#     print(y)
#     print(net(d.cuda()))
#     break


