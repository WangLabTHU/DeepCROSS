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


class PredictorNet(nn.Module):
    def __init__( self, seq_len=165, conv_hidden=64, nb_layers=6):
        super(PredictorNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=conv_hidden, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(conv_hidden)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm1 = nn.LSTM(input_size=conv_hidden, hidden_size=conv_hidden, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=conv_hidden*2, hidden_size=conv_hidden, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=conv_hidden*2, hidden_size=conv_hidden, bidirectional=True)
        
        self.latent_dim = conv_hidden * 2
        self.seq_length = int(seq_len / 2)
        
        self.denseblock1 = DenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock1 = TransitionLayer(conv_hidden*2 + 32*nb_layers)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        
        self.denseblock2 = DenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock2 = TransitionLayer(self.latent_dim)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        self.denseblock3 = DenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock3 = TransitionLayer(self.latent_dim)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        self.denseblock4 = DenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.seq_length = int(self.seq_length / 2)
        
        self.linear = nn.Linear( in_features=self.latent_dim * self.seq_length, out_features=1 )
        
    def forward(self,x):
        x = self.bn1( self.conv1(x) )
        x = self.pool1( self.relu1(x) ) # [batch_size, conv_hidden, 83]
        
        x = x.permute(2,0,1)
        x,_ = self.lstm1(x) # [83, batch_size, conv_hidden*2]
        x,_ = self.lstm2(x) # [83, batch_size, conv_hidden*2]
        x,_ = self.lstm3(x) # [83, batch_size, conv_hidden*2]
        x = x.permute(1,2,0)# [batch_size, conv_hidden*2, 83], length = np.floor((165 + 2 * 1 - 1 - 2)/2 + 1)
        
        x = self.denseblock1(x) # [batch_size, conv_hidden*2 + 32*6, 83]
        x = self.transblock1(x) # [batch_size, (conv_hidden*2 + 32*6)/2, (83/2)]       
        x = self.denseblock2(x)
        x = self.transblock2(x)
        x = self.denseblock3(x)
        x = self.transblock3(x)
        x = self.denseblock4(x) # [batch_size, latent_dim, seq_length] = [batch_size, 376, 10]
        
        x = self.avgpool1(x)
        x = self.flatten(x) # [batch_size, 376 x 5=1880]
        x = self.linear(x) # [batch_size, 1]
        return x


class PredictorNet_language:
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
      
        self.model = PredictorNet(seq_len=length)
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

'''
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
'''

def plot_reg(label_real, label_pred, plot_path, c1="indianred"):
    
    def r(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value 
    
    r_test = r(label_real, label_pred)
    testdata = pd.DataFrame({'pred': label_pred, 'label': label_real})

    sns.set_style("darkgrid")
    font = {'size' : 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    # c1 = "indianred"
    # c2 = 'mediumpurple'
    g = sns.JointGrid(x='label', y="pred", data=testdata, space=0, ratio=6, height=7)
    g.plot_joint(plt.scatter,s=20, color=c1, alpha=1, linewidth=0.4, edgecolor='black')
    f = g.fig
    ax = f.gca()
    
    ax.text(x=.71, y=0.03,s='r: ' + str(round(r_test, 2)), transform=ax.transAxes) # size
    g.plot_marginals(sns.kdeplot,fill=r_test, **{'linewidth':2, 'color':c1})
    g.set_axis_labels('NM2018', 'Pred', **{'size':18}) # **{'size':22}
    f = g.fig
    plt.savefig(plot_path)

def plot_boxplot(plot_path, barplot_data):
    font = {'size' : 8}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (20,20), dpi = 900)

    ax = sns.boxplot( x="label", y="prediction",  data=barplot_data,  boxprops=dict(alpha=.9), hue="species", # hue_order = hue_order,
                      fliersize=1, flierprops={"marker": 'x'}, palette=["cornflowerblue", "indianred"]) # # palette="viridis_r"
    h,_ = ax.get_legend_handles_labels()

    plt.xticks(rotation=270)
    
    ax.set_xlabel('Controlling', fontsize=10)
    ax.set_ylabel('Predictions', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)


if __name__ == "__main__":
    
    '''
    step1: loading dataset
    '''
    df_nm2018 = pd.read_csv("../dataset/AAE_supervised/calibration_nm2018.csv")
    seqs =  list(df_nm2018.loc[:,"seqs"])
    expr_ec = list(df_nm2018.loc[:,"EC(2018)"])
    expr_pa = list(df_nm2018.loc[:,"PA(2018)"])
    
    write_fa("../dataset/PredNet_round1/seqs.txt", seqs)
    write_exp("../dataset/PredNet_round1/expr_ec.txt", expr_ec)
    write_exp("../dataset/PredNet_round1/expr_pa.txt", expr_pa)
    
    '''
    step2: training models
    '''
    
    model = PredictorNet_language(model_name = "pred_ec", length=165)
    dataset = "../dataset/PredNet_round1/seqs.txt"
    labels_ec  = "../dataset/PredNet_round1/expr_ec.txt"
    save_path = "./check/"
    model.train(dataset=dataset,labels=labels_ec,savepath=save_path)
    
    model = PredictorNet_language(model_name = "pred_pa", length=165)
    dataset = "../dataset/PredNet_round1/seqs.txt"
    labels_pa  = "../dataset/PredNet_round1/expr_pa.txt"
    save_path = "./check/"
    model.train(dataset=dataset,labels=labels_pa,savepath=save_path)
    
    '''
    step3: model evaluation
    '''
    model = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec = "./check/pred_ec/checkpoint.pth"
    seqs_ec = open_fa("../dataset/PredNet_round1/seqs.txt")
    label_ec = open_exp("../dataset/PredNet_round1/expr_ec.txt", operator="direct")
    pred_ec = model.predict_input(model_path=model_ec, inputs=seqs_ec, mode="data")
    
    model = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa = "./check/pred_pa/checkpoint.pth"
    seqs_pa = open_fa("../dataset/PredNet_round1/seqs.txt")
    label_pa = open_exp("../dataset/PredNet_round1/expr_pa.txt", operator="direct")
    pred_pa = model.predict_input(model_path=model_pa, inputs=seqs_pa, mode="data")
    
    plot_reg(label_ec, pred_ec, "../results/prediction_round1/prednet_pcc_nm2018_ec.png",c1="mediumpurple")
    plot_reg(label_pa, pred_pa, "../results/prediction_round1/prednet_pcc_nm2018_pa.png",c1="indianred")
    
    '''
    step4: boxplot of Lib-1
    '''
    
    df_overlap = pd.read_csv("../dataset/AAE_supervised/calibration_mpra_20240201.csv")
    seqs_2024 = list(df_overlap.loc[:,"seqs"])
    label_2024 = list(df_overlap.loc[:,"tags"])
    
    model = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec = "./check/pred_ec/checkpoint.pth"
    pred_ec = model.predict_input(model_path=model_ec, inputs=seqs_2024, mode="data")
    
    model = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa = "./check/pred_pa/checkpoint.pth"
    pred_pa = model.predict_input(model_path=model_pa, inputs=seqs_2024, mode="data")
    
    pred_list = pred_ec + pred_pa
    label_list = label_2024 + label_2024
    spe_list = ["EC"] * len(label_2024) + ["PA"] * len(label_2024)
    
    df_2024 = pd.DataFrame({"prediction": pred_list, "label": label_list, "species": spe_list})
    plot_boxplot("../results/prediction_round1/boxplot_total_pred_2024.png", df_2024)

    '''
    step5: prediction of Lib-1
    overlap tags: EConly, PAonly, ECPA, bothNo
    '''

    
    df_lib1 = pd.read_csv("../dataset/AAE_supervised/calibration_mpra_20240201.csv")
    df_lib1 = df_lib1[~df_lib1['tags'].isin(['EConly', 'PAonly', 'ECPA', 'bothNo'])]
    
    seqs = list(df_lib1.loc[:,"seqs"])
    label_ec = list(df_lib1.loc[:,"EC(2024)"])
    label_pa = list(df_lib1.loc[:,"PA(2024)"])
    
    model = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec = "./check/pred_ec/checkpoint.pth"
    pred_ec = model.predict_input(model_path=model_ec, inputs=seqs, mode="data")
    
    model = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa = "./check/pred_pa/checkpoint.pth"
    pred_pa = model.predict_input(model_path=model_pa, inputs=seqs, mode="data")
    
    plot_reg(label_ec, pred_ec, "../results/prediction_round1/prednet_pcc_lib1_ec.png",c1="mediumpurple")
    plot_reg(label_pa, pred_pa, "../results/prediction_round1/prednet_pcc_lib1_pa.png",c1="indianred")
    
    
    