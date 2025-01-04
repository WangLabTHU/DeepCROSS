
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

from collections import OrderedDict

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

class TransitionLSTM(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLSTM, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([]))
        self.model1.add_module('lstm1', nn.LSTM(input_size=num_input_features, hidden_size=num_output_features,
                                         num_layers=1, bias=True, batch_first=True, bidirectional=True))

    def forward(self, x):
        x, (h_n, c_n) = self.model1(x)
        return x
    
class DenseLayerLSTM(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayerLSTM, self).__init__()
        #self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        #self.add_module('relu1', nn.ReLU(inplace=True)),
        self.model1 = nn.Sequential(OrderedDict([]))
        self.model1.add_module('lstm1', nn.LSTM(input_size=num_input_features, hidden_size=bn_size*growth_rate,
                                                       num_layers=3, bias=True, batch_first=True, bidirectional=True))
        #self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        #self.add_module('relu2', nn.ReLU(inplace=True)),
        self.model2 = nn.Sequential(OrderedDict([]))
        self.model2.add_module('lstm2', nn.LSTM(input_size=2 * bn_size * growth_rate, hidden_size=growth_rate,
                                         num_layers=3, bias=True, batch_first=True, bidirectional=True))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features, (h_n, c_n) = self.model1(x)
        new_features, (h_n, c_n) = self.model2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 2)

class DenseBlockLSTM(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlockLSTM, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([]))
        self.num_layers = num_layers
        for i in range(num_layers):
            self.model1.add_module('denselayer%d' % (i + 1), DenseLayerLSTM(num_input_features + 2 * i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        x = self.model1(x)
        return x

class DenseConnectedLSTM(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=64, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseConnectedLSTM, self).__init__()

        # First convolution
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features = nn.Sequential(OrderedDict([]))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlockLSTM(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + 2 * num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLSTM(num_input_features=num_features, num_output_features=num_features // 4)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = 2 * (num_features // 4)

        # Final batch norm
        self.fb_norm = nn.Sequential(OrderedDict([]))
        self.fb_norm.add_module('norm5', nn.BatchNorm1d(num_features))
        # Linear layer
        self.ratio = nn.Linear(int(length) * num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features0 = self.features0(x)
        features0 = features0.permute(0, 2, 1)
        features1 = self.features(features0)
        features1 = features1.permute(0, 2, 1)
        features1 = self.fb_norm(features1)
        out = F.relu(features1, inplace=True)
        out = F.avg_pool1d(out, kernel_size=7, stride=1, padding=3).view(out.size(0), -1)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out


class DenseConnectedLSTM_language:
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
      
        self.model = DenseConnectedLSTM(input_length=length)
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
