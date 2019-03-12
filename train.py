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
import math
import random
from model import DenseNet, ResNet
from progress.bar import Bar
import json

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

loss_factor = 2

def loss_fn(net, batch_in, currency_select):
    d,y = batch_in
    
    d = (d.reshape(-1, d.shape[2], d.shape[3], d.shape[4]).float())
    y = (y.reshape(-1, y.shape[2], y.shape[3]).float())
    currency_count = y.shape[1]
    # print(y.shape)
    y = (y.reshape([-1,currency_count]))[:, currency_select].cuda()
    # y = torch.cat([y[:, 0].unsqueeze(1), y[:, currency_select].unsqueeze(1)], 1)
    # print(y)
    # print(y.shape)
    saved = y
    # print(saved.shape)
    # y = ((y) ** -1) ** loss_factor
    d = d.cuda()
    # print(d.shape)
    x = net(d).squeeze()

    # print(x)

    # print(x.shape)
    # print(y.shape)
    # print(x)
    # print(saved)
    #x = (x ** -1) ** loss_factor
    
    # print(y.shape)
    # print(x.shape)
    # print(x*saved)
    # readable_loss = (x * saved).sum().sum()
    
    # readable_loss /= y.shape[0]
    # print(readable_loss.item())
    # print(readable_loss)
    # z = x * y
    # loss = z.sum().sum()
    # loss = loss/y.shape[0]

    # p = y.clone()
    # p[p >= 1] = 1
    # p[p < 1] = 0
    # best = -((p * y) - p).sum()
    # print("Best {}".format(best.item()))
    loss = -((x * y) - x).sum()
    # print("Done {}".format(loss.item()))

    # readable_loss = 1
    #print(loss)
    del y
    del d
    return loss

def test(model, test_loader, epoch, test_size, currency_select):
        model.eval()
        loss = 0
        raw_loss = 0.0
        j = 0
        for i,b in enumerate(test_loader, 0):
            j = i
            rl = loss_fn(model, b, currency_select)
            raw_loss += rl.item()
            # loss += readable_loss.item() * 10000
        print('Epoch {} test loss: {}'.format(epoch+1, raw_loss/(j+1)))
        return raw_loss

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
    def train_test_split(self, test_size=0.1, subset=1):
        full = list(range(self.__len__()))
        random.shuffle(full)
        full = full[:int(subset * len(full))]
        train = full[int(test_size * len(full)):]
        test = full[:int(test_size * len(full))]
        return train, test
        


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('cuda_dev', type=int,
    #                     help='CUDA Device ID')
    #args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_dev)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(input('CUDA Device:'))

    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    random.seed(77)
    np.random.seed(69)
    torch.manual_seed(3)
    
    global loss_factor
    loss_factor = int(input('Loss factor:'))
    drop_in = float(input('Drop rate:'))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #data = CurrencyDataset('./Processed')
    data = CurrencyDataset('./Processed/')
    train_indices, test_indices = data.train_test_split(subset=1)
    print('Train size: {}'.format(len(train_indices)))
    print('Test size: {}'.format(len(test_indices)))
    currency_count = 1
    #net = ResNet([3,8,36,3], currency_count)
    
    net = DenseNet(currency_count, reduce=True, layer_config=(4, 8, 16, 12), growth_rate=32, init_layer=64, drop_rate=drop_in)
    # net = DenseNet(currency_count, reduce=True, layer_config=(2,4,4,2), growth_rate=48, init_layer=96, drop_rate=drop_in)
    net = nn.DataParallel(net)
    net.cuda()
    net.train()
    train_batch = 300
    test_batch = 20
    dataloader = DataLoader(data,batch_size=train_batch,pin_memory=True, sampler=SubsetRandomSampler(train_indices), num_workers=4)
    test_loader = DataLoader(data,batch_size=test_batch,pin_memory=True, sampler=SubsetRandomSampler(test_indices), num_workers=4)
    #dataloader = DataLoader(data,batch_size=50,pin_memory=True,shuffle=True)

    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.LBFGS(net.parameters(), history_size=200)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    currency_select = 2

    begin = time.time()
    for epoch in range(100):
        epoch_time = time.time()
        epoch_loss = 0.0
        net.train()
        j = 0
        pbar = Bar('Epoch ' + str(epoch+1), max=len(train_indices)//train_batch, suffix='%(percent)d%% %(eta)d seconds left.')
        for i,batch in enumerate(dataloader, 0):
            j = i
            # def closure():
            #     optimizer.zero_grad()
            #     loss = loss_fn(net, batch, currency_select)
            #     # print(loss)
            #     # print(readable_loss * 10000)
            #     # global epoch_loss
            #     # epoch_loss += readable_loss.item() * 10000
            
            #     return loss
            optimizer.zero_grad()
            loss = loss_fn(net, batch, currency_select)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.next()
            
        print('\nEpoch {} training gain: {}'.format(epoch+1, epoch_loss/(j+1)))
        with torch.no_grad():
            running_loss = test(net, test_loader, epoch, len(test_indices), currency_select)
        # running_loss = test(net, test_loader, epoch, len(test_indices))
        #scheduler.step(running_loss)
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Time taken for epoch: {} minutes\n'.format((time.time()-epoch_time)/60))
        pbar.finish()
    end = time.time()
    print("FINAL TIME: {}".format(end-begin))

if __name__ == "__main__":
    main()
    
