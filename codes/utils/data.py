'''
data helper
charmap: {0: 'T', 1: 'C', 2: 'G', 3: 'A'}
'''

import os, sys
import numpy as np
import pandas as pd
from gpro.utils.utils_predictor import seq2onehot, open_fa, open_exp

def load_seq_data(datafile):
    seqs = open_fa(datafile)
    oh = seq2onehot(seqs, len(seqs[0]))
    return oh, seqs

def load_supervise_data(supervise_file, nbin, batch_size, mode, batch_dict_name = ['data_train', 'label_train','data_valid','label_valid']):
    data = SupervisedData(supervise_file, nbin, mode, batch_size, batch_dict_name)
    return data

def split_data(data, r=0.9):
    idx = np.random.permutation(data.shape[0])
    n = int(data.shape[0] * r)
    idx_train, idx_val = idx[:n], idx[n:]
    data_train, data_val = data[idx_train], data[idx_val]
    return data_train, data_val

def split_data_supervised(onehot, label, r=0.9):
    idx = np.random.permutation(onehot.shape[0])
    n = int(onehot.shape[0] * r)
    idx_train, idx_val = idx[:n], idx[n:]
    onehot_train, onehot_val = onehot[idx_train], onehot[idx_val]
    label_train, label_val = label[idx_train], label[idx_val]
    return onehot_train, label_train, onehot_val, label_val


def onehot2seq(onehot):
    ref = {0: 'T', 1: 'C', 2: 'G', 3: 'A'}
    seq_list = []
    for item in onehot:
        seq = ''
        for letter in item:
            idx = np.where(letter == np.amax(letter))[0]
            if ( len(idx) ==1 ):
                letter = int(idx)
            else:
                letter = np.random.choice(idx)
            seq = seq + ref[letter]
        if seq != '':
            seq_list.append(seq)
    return seq_list

'''
mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (1, 1): 4,
        (1, 2): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8
    }
'''

def map_labels(label, nbin):
    tag = label[0] * (nbin) + label[1]
    return tag


'''
supervised_data class, for the semi-supervised training process
'''
class SupervisedData():
    def __init__(self, supervise_file, nbin, mode, batch_size=64, batch_dict_name=None, shuffle=True): 
        self._shuffle = shuffle
        self.nbin = nbin
        self._batch_dict_name = batch_dict_name

        self._batch_size = batch_size
        self._load_files(supervise_file, mode) 
        self._seq_id_train = 0
        self._seq_id_test = 0
    
    
    def _load_files(self, supervise_file, mode):
        
        if (mode == "round1"):
            tags = ["EC(2018)", "PA(2018)"]
        elif (mode == "round2"):
            tags = ["EC(2024)", "PA(2024)"]
        
        supervise_file_path = supervise_file
        supervise_data = pd.read_csv(supervise_file_path)
        
        exp_all=[]
        for i in range(len(supervise_data)):
            exp_all.append( [ supervise_data.loc[i,tags[0]], supervise_data.loc[i,tags[1]] ] )
        exp_all=sum(exp_all,[])
        
        _, binlist = np.histogram(exp_all, bins=3)
        binlist[-1]=binlist[-1]+1
        binlist = np.array(binlist)
    
        seqs_list, label_list=[], []
        expression_list = []
        
        for i in range(len(supervise_data)):
            EC_1=supervise_data.loc[i,tags[0]]
            PA_1=supervise_data.loc[i,tags[1]]
            index_EC=np.where(EC_1>=binlist)[0][-1]
            index_PA=np.where(PA_1>=binlist)[0][-1]
            
            seq = supervise_data.loc[i,"seqs"]
            seqs_list.append(seq)
            label_list.append([index_EC,index_PA])
            expression_list.append([EC_1, PA_1])
        
        ## (0410) mapping: from paired data to single description
        onehot_list = seq2onehot(seqs_list, len(seqs_list[0]))
        label_list = [map_labels(label, self.nbin) for label in label_list]
        
        self.onehot_list = np.array(onehot_list)
        self.label_list = np.array(label_list) 
        self.seqs_list= np.array(seqs_list)
        
        ## expression levels for calibration
        self.binlist = np.array(binlist)
        self.expression_list = np.array(expression_list)
        
        self.onehot_list_train, self.label_list_train, self.onehot_list_test, self.label_list_test = split_data_supervised(self.onehot_list, self.label_list)
        self.datanum_train=self.onehot_list_train.shape[0] # (8153, 165, 4), (8153, )
        self.datanum_test=self.onehot_list_test.shape[0] # (906, 165, 4), (906, )
        self._shuffle_files(labeluse=True)

    def size(self,labeluse=False):
        return self.onehot_list.shape[0]
    
    def _shuffle_files(self,train_test_flag='train',labeluse=False):
        if self._shuffle:
            if train_test_flag=='train':
                idxs = np.arange(self.datanum_train)
                np.random.shuffle(idxs)
                self.onehot_list_train = self.onehot_list_train[idxs]
                if labeluse:
                    self.label_list_train = self.label_list_train[idxs]
            else:
                idxs = np.arange(self.datanum_test)
                np.random.shuffle(idxs)
                self.onehot_list_test = self.onehot_list_test[idxs]
                if labeluse:
                    self.label_list_test = self.label_list_test[idxs]

    def next_batch_dict(self, labeluse=False):
        batch_data = self.next_batch(labeluse=labeluse) # (64, 165, 4) (64, ) (64, 165, 4) (64, )
        data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
        return data_dict
    
    def next_batch(self,labeluse=False):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        
        #train
        start_train = self._seq_id_train
        self._seq_id_train += self._batch_size
        end_train = self._seq_id_train
        batch_data_train = self.onehot_list_train[start_train:end_train]
        batch_label_train = self.label_list_train[start_train:end_train]
        
        self._shuffle_files(train_test_flag='train',labeluse=labeluse)
        if self._seq_id_train + self._batch_size > self.datanum_train:
            self._seq_id_train = 0
        
        #test
        start_test = self._seq_id_test
        self._seq_id_test += self._batch_size
        end_test = self._seq_id_test
        batch_data_test = self.onehot_list_test[start_test:end_test]
        batch_label_test = self.label_list_test[start_test:end_test]

        self._shuffle_files(train_test_flag='test',labeluse=labeluse)
        if self._seq_id_test + self._batch_size > self.datanum_test:
            self._seq_id_test = 0

        return [batch_data_train, batch_label_train,batch_data_test, batch_label_test]



if __name__ == "__main__":
    supervise_file = "../dataset/AAE_supervised/BS_EC_PA_exp_bin_3_train80_originnobin.npy"
    nbin = 3
    batch_dict_name = ['data_train', 'label_train','data_valid','label_valid']
    supervise = SupervisedData(supervise_file, nbin, 64, batch_dict_name)
    
    tmp = supervise.next_batch_dict()
    print(tmp['data_train'].shape, tmp['label_train'].shape)
    print(tmp['data_valid'].shape, tmp['label_valid'].shape)
    
    

    
    
    