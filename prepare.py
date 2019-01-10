
#%%
import pandas as pd
import os
import itertools
import numpy as np
import gc
import math
import time
from multiprocessing import Process
gc.enable()


#%%
class Market:
    def __init__(self, currencies, data_path, processor):
        self.path = data_path
        self.currencies = currencies
        self.pairs = list(itertools.permutations(currencies,2))
        self.reference_currency = 'USD'
        self.proc_count = processor
        
    def import_file(self):
        df = {}
        data_path = os.path.abspath(self.path)
        available_files = os.listdir(data_path)
        
        for pair_tuple in self.pairs:
            pair = pair_tuple[0] + pair_tuple[1]
            if (pair + '.csv') in available_files:
                if os.path.isfile(os.path.join(data_path, pair + '.csv')):
                    
                    df[pair] = pd.read_csv(os.path.join(data_path, pair + '.csv'), delimiter='\t', 
                                            usecols=['Timestamp', 'Open', 'High', 'Low', 'Close'])
                else:
                    continue
        self.data = df
    
    def process_time_period(self, timePeriod, index, size):
        allPrices = np.zeros(shape=(size, 3, timePeriod, len(self.currencies)))
        allRates = np.zeros(shape=(size, len(self.currencies), 1))
        dimensions = ['Open', 'High', 'Low']
        m = 0
        for currency in self.currencies:
            if currency + self.reference_currency in self.data.keys():
                pair = currency + self.reference_currency
            elif self.reference_currency + currency in self.data.keys():
                pair = self.reference_currency + currency
            elif self.reference_currency == currency:
                for i in range(size):
                    allPrices[i, :, :, m] = 1
                    allRates[i, m, 0] = 1
                m += 1
                continue
            else:
                raise ValueError('Wrong currency parameter.')
            batchValues = self.data[pair].iloc[index : index + timePeriod + size, 1:4].values
            for i in range(size):
                movement = batchValues[i:timePeriod+i]
                refVal = movement[-1][0]
                if (refVal == 0):
                    raise ValueError(index)
                movement = movement / refVal
                movement[movement <= 0.0] = 0.1
                movement[movement >= 5.0] = 1.0
                nextPrice = batchValues[timePeriod+i][0]
                rate = nextPrice / refVal
                if (rate == 0):
                    raise ValueError(index)
                if (pair[0:3] == 'USD'):
                    movement **= -1
                    allPrices[i, :, :, m] = np.transpose(movement)
                    allRates[i, m, 0] = 1/rate
                else:
                    allPrices[i, :, :, m] = np.transpose(movement)
                    allRates[i, m, 0] = rate
            m += 1
        return (allPrices, allRates)
    
    def prepare_data(self, batch_size, period_size, reset=False, count=-1, start=1):
        self.batch_path = os.path.abspath("./Processed/Batches")
        self.label_path = os.path.abspath("./Processed/Labels")
        self.batch_size = batch_size
        self.period_size = period_size
       
        if not os.path.exists(self.batch_path):
            os.makedirs(self.batch_path)
        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)
        if reset:
            for folder in (self.batch_path, self.label_path):
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        raise ValueError('Error encountered when deleting file!')
        now = time.time()
        min_size = math.inf
        for pair in self.data.items():
            if len(pair[1].index) < min_size:
                min_size = len(pair[1].index)
        if count == -1:
            self.batch = int(math.floor(min_size / self.batch_size)) - 1
        else:
            self.batch = count
        name = 0
        each = self.batch // self.proc_count
        processes = []
        print(each)
        print(self.proc_count)
        for i in range(self.proc_count):
            p = Process(target=self.export_range, args=((list(range(i*each, (i+1)*each)),)))
            processes.append(p)
            p.start()
        #for i in range(start,self.batch + start):
         #   self.export_batch(i*self.batch_size,name)
          #  name += 1
        for p in processes:
            p.join()
        later = time.time()
        print((later-now)/60)

    def export_range(self, index):
        for i in index:
            self.export_batch(i, i)

    def export_batch(self, index, name):
        try:
            (movements, rates) = self.process_time_period(self.period_size, index, self.batch_size)
            if self.batch_size == 1:
                movements = np.squeeze(movements,axis=0)
                rates = np.squeeze(rates,axis=0)
            np.save((os.path.join(self.batch_path, "Batch_" + str(name))), movements)
            np.save((os.path.join(self.label_path, "Label_" + str(name))), rates)
        except:
            raise ValueError(str(index) +' and '+ str(name))


#%%
market = Market(['TRY','USD', 'EUR', 'NZD', 'GBP'],'./Processed', 16)
market.import_file()
market.prepare_data(1,50,reset=True)



