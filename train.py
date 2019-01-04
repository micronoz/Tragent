
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
from model import DenseNet, ResNet


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('cuda_dev', type=int,
                    help='CUDA Device ID')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_dev)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:
def loss_fn(net, batch_in):
    d,y = batch_in
    d = d.float()
    y = y.float()
    
    y[y <= 0.0] = 1.0
    y[y > 3.0] = 1.0
    saved = y
    #y = torch.tan(-y - 1)
    #y = y ** -1
    y = (y ** -1) ** 2
#         d = test_X
#         y = test_y
    d = d.cuda()
    x = net(d)
    y = (y.reshape([-1,currency_count])).cuda()
#         print(x.shape)
#         print(y.shape)
    readable_loss = torch.mul(x,saved.reshape([-1,currency_count]).cuda()).sum(dim=1)
    s = readable_loss.shape[0]
    readable_loss = readable_loss.sum()/s
    z = torch.mul(x,y)
    z = z.sum(dim=1)
    loss = z.sum()
    loss /= z.shape[0]
    del y
    del d
    return loss,readable_loss

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


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


# In[4]:


torch.backends.cudnn.deterministic = False
torch.manual_seed(991)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
data = CurrencyDataset('./Processed')
device = 'cuda:0'
currency_count = 5
#net = DenseNet(currency_count,growth_rate=1)
#net = Net()
net = ResNet([3,8,36,3], currency_count)
print(net)
# net = nn.DataParallel(net,device_ids=[0])
net.cuda()
# net = net.half()
# for layer in net.modules():
#     if isinstance(layer, nn.BatchNorm2d):
#         layer.float()
# net = net.to(device)
net.train()
dataloader = DataLoader(data,batch_size=50,pin_memory=True, sampler=SubsetRandomSampler(list(range(5000,15000))))
test_loader = DataLoader(data,batch_size=5,pin_memory=True, sampler=SubsetRandomSampler(list(range(3400,3500))))
#dataloader = DataLoader(data,batch_size=50,pin_memory=True,shuffle=True)


# In[5]:


optimizer = optim.Adam(net.parameters(), lr=0.01)


# In[6]:

def test(model, test_loader, epoch):
    model.eval()
    loss = 0
    j = 0
    for i,b in enumerate(test_loader, 0):
        j = i
        _, readable_loss = loss_fn(model, b)
        loss += readable_loss.item() * 10000
    print('Epoch {} test gain: {}\n'.format(epoch+1, loss/(j+1)))


net.apply(weight_reset)
# test_X, test_y = iter(dataloader).next()
# print(test_X)
# print(test_y)
print_every = 16
begin = time.time()
for epoch in range(200):
    epoch_loss = 0.0
    running_loss = 0.0
    net.train()
    j = 0
    for i,batch in enumerate(dataloader, 0):
        j = i
        optimizer.zero_grad()
        loss, readable_loss = loss_fn(net, batch)
        epoch_loss += readable_loss.item() * 10000
        
        # if i % print_every == print_every - 1:
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / print_every))
        #     print('Time elapsed: {} minutes'.format((time.time() - begin)/60))
        #     running_loss = 0.0

        loss.backward()
        optimizer.step()
    print('Epoch {} training gain: {}'.format(epoch+1, epoch_loss/(j+1)))
    test(net, test_loader, epoch)
end = time.time()
print("FINAL TIME: {}".format(end-begin))



