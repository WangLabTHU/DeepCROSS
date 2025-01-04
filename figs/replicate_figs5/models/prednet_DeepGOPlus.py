
import os, sys
import numpy as np
import pandas as pd
import time, datetime
import scipy.stats as stats
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pickle
import random
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from gpro.utils.utils_predictor import EarlyStopping, seq2onehot, open_fa, open_exp

from utils.data import load_seq_data, load_supervise_data, split_data
from utils.data import *
from utils.module import *
from utils.function import *

from gpro.utils.base import write_fa, write_seq, write_exp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

class SequenceData(Dataset):
  def __init__(self,data, label):
    self.data = data
    self.target = label
  
  def __getitem__(self, index):
    return self.data[index], self.target[index]
    
  def __len__(self):
    return self.data.size(0)
  
  def __getdata__(self):
    return self.data, self.target

class TestData(Dataset):
    def __init__(self,data):
        self.data = data
  
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return self.data.size(0)
    
    def __getdata__(self):
        return self.data



class deepGoPlus(nn.Module):

    def __init__(self,
                 input_nc=4,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True, n_blocks=3, padding_type='reflect', seqL = 100, max_kernels=79, in_nc=512, hidden_dense=0):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(deepGoPlus, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(6, max_kernels, 2)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        for i in range(hidden_dense):
            self.fc.append(nn.Linear(len(kernels)*in_nc, len(kernels)*in_nc))
        self.fc.append(nn.Linear(len(kernels)*in_nc, 1))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            if x.size(0) > 1:
                exec("x_list.append(torch.squeeze(x_i))")
            else:
                exec("x_list.append(torch.squeeze(x_i).reshape([1, -1]))")
        x1 = torch.cat(tuple(x_list), dim=1)
        x2 = self.fc(x1)
        output = x2.squeeze(-1)
        return output




class deepGoPlus_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "predictor",
                 epoch = 200,
                 patience = 50,
                 log_steps = 10,
                 save_steps = 20,
                 exp_mode = "direct"
                 ):
      
        self.model = deepGoPlus(seqL=length)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.seq_len = length
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
        self.exp_mode = exp_mode

    def train(self, dataset, labels, savepath, ratio=0.7):
      
        self.dataset = dataset
        self.labels = labels
        self.checkpoint_root = savepath
      
        filename_sim = self.checkpoint_root + self.model_name
        
        if not os.path.exists(filename_sim):
            os.makedirs(filename_sim)
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, 
                                       path=os.path.join(filename_sim, 'checkpoint.pth'), stop_order='max')
        
        total_feature = open_fa(self.dataset)
        total_feature = seq2onehot(total_feature, self.seq_len)
        total_label = open_exp(self.labels, operator=self.exp_mode)
        total_feature = torch.tensor(total_feature, dtype=float) # (sample num,length,4)
        total_label = torch.tensor(total_label, dtype=float) # (sample num)
        
        total_length = int(total_feature.shape[0])
        r = int(total_length*ratio)
        train_feature = total_feature[0:r,:,:]
        train_label = total_label[0:r]
        valid_feature = total_feature[r:total_length,:,:]
        valid_label = total_label[r:total_length]
        
        train_dataset = SequenceData(train_feature, train_label)
        train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)
        valid_dataset = SequenceData(valid_feature, valid_label)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=self.batch_size, shuffle=True)
        
        train_log_filename = os.path.join(filename_sim, "train_log.txt")
        train_model_filename = os.path.join(filename_sim, "checkpoint.pth")
        print("results saved in: ", filename_sim)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.HuberLoss(reduction='mean')
        
        for epoch in tqdm(range(0,self.epoch)):
            model.train()
            train_epoch_loss = []
            for idx,(feature,label) in enumerate(train_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                optimizer.zero_grad()
                loss = criterion(label.float(),outputs.flatten())
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

            model.eval()
            valid_exp_real = []
            valid_exp_pred = []
            for idx,(feature,label) in enumerate(valid_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                valid_exp_real += label.float().tolist()
                valid_exp_pred += outputs.flatten().tolist()
            coefs = np.corrcoef(valid_exp_real,valid_exp_pred)
            coefs = coefs[0, 1]
            test_coefs = coefs
            
            print("real expression samples: ", valid_exp_real[0:5])
            print("pred expression samples: ", valid_exp_pred[0:5])
            print("current coeffs: ", test_coefs)
            cor_pearsonr = pearsonr(valid_exp_real, valid_exp_pred)
            print("current pearsons: ",cor_pearsonr)
            
            ## Early Stopping Step
            early_stopping(val_loss=test_coefs, model=self.model)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
            
            if (epoch%self.log_steps == 0):
                to_write = "epoch={}, loss={}\n".format(epoch, np.average(train_epoch_loss))
                with open(train_log_filename, "a") as f:
                    f.write(to_write)
            if (epoch%self.save_steps == 0):
                torch.save(model.state_dict(), train_model_filename)
    
    def predict(self, model_path, data_path):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        path_seq_save =  '{}/seqs.txt'.format(model_path)
        path_pred_save = '{}/preds.txt'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        test_feature = open_fa(data_path)
        test_seqs = test_feature
        
        test_feature = seq2onehot(test_feature, seq_len)
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        test_exp_pred = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            test_exp_pred += pred
        
        ## Saving Seqs
        f = open(path_seq_save,'w')
        i = 0
        while i < len(test_seqs):
            f.write('>' + str(i) + '\n')
            f.write(test_seqs[i] + '\n')
            i = i + 1
        f.close()
        
        ## Saving pred exps
        f = open(path_pred_save,'w')
        i = 0
        while i < len(test_exp_pred):
            f.write(str(np.round(test_exp_pred[i],2)) + '\n')
            i = i + 1
        f.close()

    def predict_input(self, model_path, inputs, mode="path"):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        if mode=="path":
            test_feature = open_fa(inputs)
            test_feature = seq2onehot(test_feature, seq_len)
        elif mode=="data":
            test_feature = seq2onehot(inputs, seq_len)
        elif mode=="onehot":
            test_feature = inputs
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        exp = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            exp += pred
        return exp





