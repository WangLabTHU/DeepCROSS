
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

import functools
import math
from torch.autograd import Variable

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

class PositionalEncoding(nn.Module): 
  "Implement the PE function." 
  def __init__(self, d_model,max_len=165,dropout=0.4): 
    super(PositionalEncoding, self).__init__() 
    self.dropout = nn.Dropout(p=dropout) 

    pe = torch.zeros(max_len, d_model) 
    position = torch.arange(0, max_len).unsqueeze(1) 
    div_term = torch.exp(torch.arange(0, d_model, 2) * \
      -(math.log(10000.0) / d_model)) 
    pe[:, 0::2] = torch.sin(position * div_term) 
    pe[:, 1::2] = torch.cos(position * div_term) 
    pe = pe.unsqueeze(0) 
    self.register_buffer('pe', pe) 

  def forward(self, x): 
    out = Variable(self.pe[:,:x.size(2)], requires_grad=False).transpose(1,2) 
    x = x + out 
    return self.dropout(x) 

class FeedForward(nn.Module):
    """Define a FeedForward block"""

    def __init__(self,conv_dim=256,kernel_size=3):
        super(FeedForward, self).__init__() 
        self.feedforward = self.build_feedforward(conv_dim=conv_dim,kernel_size=kernel_size)

    def build_feedforward(self,conv_dim,kernel_size=3):
        feedforward = []
        feedforward+=[nn.BatchNorm1d(conv_dim),#64 or 128
                nn.Conv1d(conv_dim, conv_dim*2, kernel_size=kernel_size, padding=1),
                nn.Dropout(0.4),
                nn.ReLU(inplace=True),
                nn.Conv1d(conv_dim*2, conv_dim, kernel_size=kernel_size, padding=1),
                nn.Dropout(0.4),]
        return nn.Sequential(*feedforward)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.feedforward(x)  # add skip connections
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim,n_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.model_dim=model_dim 
        self.n_head=n_head 
        self.head_dim = self.model_dim // self.n_head #32

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head) #256,256

        self.linear_final=nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layernorm = nn.LayerNorm(self.model_dim, eps=1e-05)


    def forward(self,qkv, mask=None): #query, key, value
        query=qkv
        key=qkv
        value=qkv
        q = self.linear_q(query) 
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size=k.size()[0]

        q_ = q.view(batch_size * self.n_head, -1, self.head_dim) #[1280, 64, 2]
        k_ = k.view(batch_size * self.n_head, -1, self.head_dim) #[1280, 64, 2]
        v_ = v.view(batch_size * self.n_head, -1, self.head_dim) #[1280, 64, 2]
        qkv_ = torch.stack([q_,k_,v_],dim=0)#[3,1280, 64, 2] #torch.cat
        context = self.scaled_dot_product_attention(qkv_,mask)#(q_, k_, v_, mask) 
        output = context.view(batch_size, -1, self.head_dim * self.n_head) 
        output = self.linear_final(output)
        output = self.dropout(output)
        output = self.layernorm(output)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature=3, attn_dropout=0.1): 
        super().__init__()
        # that is, sqrt dk
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, qkv, mask=None):#q, k, v,
        q=qkv[0] #[1280, 64, 2]
        k=qkv[1] #[1280, 64, 2]
        v=qkv[2] #[1280, 64, 2]
        attn = torch.matmul(q / self.temperature,k.transpose(1, 2)) #(2,3)  

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=3):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=kernel_size)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=3):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_block(x)
        out = x + out
        return out

#Enformer
class Enformer(nn.Module): #
    def __init__(self,
                 input_nc=4,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True,
                 n_blocks=3,
                 padding_type='reflect',
                 seqL=100):
        assert (n_blocks >= 0)
        super(Enformer, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        self.n_blocks = n_blocks
        self.conv_dim = 256
        #1.Stem
        model = [nn.ReflectionPad1d((3, 2)),
                 nn.Conv1d(input_nc, self.conv_dim, kernel_size=7, padding=3),
                 ResnetBlock(self.conv_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, #64
                                  use_bias=use_bias, kernel_size=3),
                 nn.MaxPool1d(kernel_size=2, stride=2)]  
        self.model = nn.Sequential(*model)
        seqT = 85 

        #3.Transformer
        self.MHA_layer = nn.Sequential(PositionalEncoding(d_model=self.conv_dim,max_len=seqT),MultiHeadAttention(model_dim=seqT,n_head=5,dropout_rate=0.4),FeedForward(conv_dim=self.conv_dim,kernel_size=3)) #64, embedding相当于是前面做过了

        #4.Outputs
        self.Pointwiseforward=nn.Sequential(nn.BatchNorm1d(self.conv_dim),
                 nn.ReLU(inplace=True),
                 nn.Conv1d(self.conv_dim, self.conv_dim*2,kernel_size=3, padding=1),
                 nn.Dropout(0.4),
                 nn.Conv1d(self.conv_dim*2, self.conv_dim*2, kernel_size=3, padding=1))
        self.Linear = nn.Sequential(nn.Linear(self.conv_dim*2*seqT, self.conv_dim*2),nn.Linear(self.conv_dim*2, 1))

    def forward(self, inputSeq):
        x = self.model(inputSeq)
        x=self.MHA_layer(x) 
        x=self.Pointwiseforward(x)
        x = x.view(x.size(0), -1)
        output = self.Linear(x)
        # output = nn.ReLU(inplace=True)(output.squeeze(-1))
        return output


class CNNwithAttention_language:
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
      
        self.model = Enformer(seqL=length)        
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
