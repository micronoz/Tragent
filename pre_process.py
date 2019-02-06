#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import zipfile as zf
import os
import shutil
import numpy as np
import itertools
from functools import reduce


# In[2]:


currencies = ['USD', 'TRY', 'GBP', 'JPY', 'EUR', 'NZD']
master_dir = os.path.abspath(input('Master path: '))
data_path = os.path.join(master_dir, 'Data')
dirs = os.listdir(data_path)
pairs = [(str(i[0]) + str(i[1])) for i in list(itertools.permutations(currencies, 2)) if 'USD' in i]
years = []
for word in dirs:
    if '20' in word:
        years.append(word)
available_currencies = set()
for year in years:
    for elements in os.listdir(os.path.join(data_path, year)):
        if elements in pairs:
            available_currencies.add(elements)


# In[3]:


class PairData:
    def __init__(self, name, tick:str):
        self.name = name
        self.data = tick
        self.month = {"01":None, "02":None,"03":None, "04":None, "05":None,
                      "06":None, "07":None, "08":None, "09":None, "10":None,
                      "11":None, "12":None}
        self.year = dict()
    def add_month(self, frame_path):
        index = frame_path.find('_20')
        year_select = frame_path[index+1 : index + 5]
        if year_select not in self.year.keys():
            self.year[year_select] = self.month.copy()
        if frame_path[index+5] != '.':
            month_select = frame_path[index + 5 : index + 7]
        else:
            month_select = "01"
            if self.year[year_select][month_select] != None:
                return
        if month_select not in self.month.keys():
            return
        self.year[year_select][month_select] = pd.read_csv(frame_path, header=None, delimiter=';',float_precision='high')
    def get_month(self, month_in, year_in):
        return self.year[year_in][month_in]
    def get_history(self):
        return self.year


# In[4]:

#process_dir = os.path.join(master_dir, 'Processed/')
process_dir = os.path.abspath('./Processed')

if not os.path.exists(process_dir):
    os.makedirs(process_dir)


# In[5]:


def unzip_files(path_name):
    files = os.listdir(path_name)
    unzip_name = os.path.join(path_name, 'Unzip')
    if not os.path.exists(unzip_name):
        os.makedirs(unzip_name)
    done_file = os.path.join(unzip_name,'Done.txt')
    if not os.path.exists(done_file):
        f = open(done_file, 'w')
        for file in files:
            if 'ASCII' in file and '.zip' in file:
                zip_file = zf.ZipFile(os.path.join(path_name, file), 'r')
                zip_file.extractall(os.path.abspath(unzip_name))
                zip_file.close()
        unzipped_files = os.listdir(os.path.abspath(unzip_name))
        tick_folder = os.path.join(unzip_name, 'Tick')
        minute_folder = os.path.join(unzip_name,'Minute')
        if not os.path.exists(tick_folder):
            os.makedirs(tick_folder)
        if not os.path.exists(minute_folder):
            os.makedirs(minute_folder)
        for file in unzipped_files:
            if '.csv' in file:
                if '_T_' in file:
                    shutil.move(os.path.join(unzip_name,file), os.path.join(tick_folder, file))
                if '_M1_' in file:
                    shutil.move(os.path.join(unzip_name, file), os.path.join(minute_folder, file))


# In[6]:


for pair in available_currencies:
    for year in years:
        path_name = os.path.join(data_path, year, pair)
        if os.path.isdir(path_name):
            unzip_files(path_name)


# In[7]:


data_frames = {}
for year in years:
    for pair in available_currencies:
        if pair not in data_frames.keys():
            data_frames[pair] = PairData(pair, 'min')
        current_pair = data_frames[pair]
        path_name = os.path.join(data_path, year, pair)
        if not os.path.isdir(path_name):
            continue
        files = os.listdir(path_name)
        unzip_name = os.path.join(path_name, 'Unzip', 'Minute')
        frames = os.listdir(unzip_name)
        for frame in frames:
            if 'DAT' in frame:
                current_pair.add_month(os.path.join(unzip_name, frame))


# In[8]:


frame_collections = dict()
for key, value in data_frames.items():
    current_pair = data_frames[key].get_history()
    concat_frames = []
    frame_collections[key] = concat_frames
    for year_key in sorted(current_pair.keys()):
        for month_key in sorted(current_pair[year_key].keys()):
            if current_pair[year_key][month_key] is not None:
                concat_frames.append(current_pair[year_key][month_key])


# In[9]:


prepared_frames = dict()
for key in frame_collections.keys():
    print(key)
    current_pair = frame_collections[key]
    df = pd.concat(current_pair)
    pd.set_option('display.max_columns', 30)
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Vol']
    df = df.drop(columns=['Vol'])
    df.reset_index(drop=True, inplace=True)
    prepared_frames[key] = df


# In[10]:


for key in prepared_frames.items():
    print(key[0],key[1].shape)


# In[11]:


data_frames = [item[1] for item in prepared_frames.items()]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Timestamp'],
                                            how='outer', sort=True), data_frames)


# In[12]:


df_key = pd.DataFrame(df_merged['Timestamp'])


# In[13]:


for datum in prepared_frames.items():
    df = pd.merge(df_key,datum[1],on=['Timestamp'],how='outer',sort=True)
    key = datum[0]
    if any(df.iloc[0].isna()):
        if key[:3] == 'USD':
            df.loc[0,['Open','High','Low','Close']] = 0
        else:
            val = df['Open'].max(axis=0)
            df.loc[0,['Open','High','Low','Close']] = val * 2
    df = df.interpolate()
    file_name = os.path.join(process_dir, key + '.csv')
    df.to_csv(file_name, sep='\t')

