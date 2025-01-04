import os, sys
import numpy as np
import pandas as pd
import time, datetime

import torch
import itertools
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pickle
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

from utils.data import load_seq_data, load_supervise_data, split_data
from utils.data import *
from utils.module import *
from utils.function import *

from evaluation import *
from prednet_r1 import plot_reg, PredictorNet_language

from gpro.utils.base import write_fa, write_seq, write_exp
from gpro.utils.utils_predictor import EarlyStopping, seq2onehot, open_fa, open_exp

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

class APN(nn.Module): # Pretraining Network for embedding_y
    def __init__(self, 
                 data_seqlen=165, data_dim=4, nbin = 3, kernel_size=3, latent_dim=128, 
                 z_dim=64, n_layers=8, batch_size=64, lamda=10, eta=0.2, penalty=0.1):
        
        super(APN, self).__init__()
    
        print("Loading pretrain dataset...\n")
        self.data_seqlen, self.data_dim = data_seqlen, data_dim
        self.kernel_size, self.latent_dim, self.z_dim, self.n_layers = kernel_size, latent_dim, z_dim, n_layers
        self.nbin, self.batch_size, self.lamda = nbin, batch_size, lamda
        self.eta, self.penalty = eta, penalty

        self.encoder = EncoderNet(self.data_dim, self.latent_dim, self.n_layers)
        self.embedding_y = ProjectionY(self.data_seqlen, self.latent_dim, self.nbin) 
    
    def forward(self, x, labels):
        self.x_input = x 
        self.x_encoded = self.encoder( x.permute(0,2,1) ) # output of E-Net, [batch_size, 512, 41]
        self.y = self.embedding_y(self.x_encoded) # output of "linear-y", [batch_size, 9]
        self.argmax_y = torch.argmax(self.y, dim=-1) # Para for get_cls_accuracy, [batch_size], single mode
        
        self.labels = labels
        self.cls_loss = self.get_cls_loss()
        
        return self.cls_loss
        
    def get_cls_loss(self):
        logits = self.y # [64,9]
        cls_loss = nn.CrossEntropyLoss()(logits, self.labels)
        return cls_loss

    def get_cls_accuracy(self): # all labels
        logits = self.argmax_y.float()
        labels = self.labels.float()
        num_correct = (labels == logits).float()
        acc = torch.mean(num_correct)
        return acc
    
    def generate_exp(self, onehot):
        x_encoded = self.encoder( onehot.permute(0,2,1) ) 
        y = self.embedding_y(x_encoded) 
        argmax_y = torch.argmax(y, dim=-1) 
        return argmax_y.tolist()
    
class AAE(nn.Module): # Adversarial Auto Encoder
    
    def __init__(self, 
                 data_seqlen=165, data_dim=4, nbin = 3, kernel_size=3, latent_dim=128, 
                 z_dim=64, n_layers=8, batch_size=64, lamda=10, eta=0.2, penalty=0.1):
        
        super(AAE, self).__init__()
        
        print("Loading dataset...\n")
        
        self.data_seqlen, self.data_dim = data_seqlen, data_dim
        self.kernel_size, self.latent_dim, self.z_dim, self.n_layers = kernel_size, latent_dim, z_dim, n_layers
        self.nbin, self.batch_size, self.lamda = nbin, batch_size, lamda
        self.eta, self.penalty = eta, penalty
        
        print('Building model...\n')
        
        """ Concate Model """
        self.Wc = nn.Parameter(torch.full((self.nbin * self.nbin, self.z_dim), 0.4)) # [9,64]
        
        self.encoder = EncoderNet(self.data_dim, self.latent_dim, self.n_layers)
        self.embedding_z = ProjectionZ(self.data_seqlen, self.latent_dim, self.z_dim)
        self.embedding_y = ProjectionY(self.data_seqlen, self.latent_dim, self.nbin) 
        self.decoder = DecoderNet(self.z_dim, self.latent_dim, self.n_layers, self.data_seqlen)

        self.discriminator_z = DiscriminatorZ(self.z_dim)
        self.discriminator_y = DiscriminatorY(self.nbin**2)
        
    def forward(self, x, labels=None):
        
        if labels is not None:
            self.labels = labels
        
        # x: [ batch_size, 165, 4] (already one-hot encoded)
        self.x_input = x # must be onehot!!!
        self.x_encoded = self.encoder( x.permute(0,2,1) ) # output of E-Net, [batch_size, 512, 41]
        self.z = self.embedding_z(self.x_encoded) # output of "linear-z", [batch_size, 64]
        
        self.y = self.embedding_y(self.x_encoded) # output of "linear-y", [batch_size, 9]
        self.argmax_y = torch.argmax(self.y, dim=-1) # Para for get_cls_accuracy, [batch_size], single mode
        self.softmax_y = F.softmax(self.y, dim=-1) # output for "softmax-y", [batch_size, 9]
        self.cluster_y = torch.matmul(self.softmax_y, self.Wc) # Para for cluster_y, [batch_size, 64]
        
        self.x_represented = torch.add(self.z, self.cluster_y) # output of Represetation, [batch_size, 64]
        self.x_decoded = self.decoder(self.x_represented) # output of G-Net, [batch_size, 165, 4]
        self.x_recon = torch.argmax(self.x_decoded, dim=-1) # reconstructed seqs
        # x_recon: [batch_size, 165]
        
        return self.x_recon
    
        '''
        if labels is not None:
            self.labels = labels
            self.cls_loss = self.get_cls_loss()
            return self.cls_loss
        else:
            self.xent_loss = self.get_xent_loss()
            self.gaussian_dnet_loss, self.gaussian_enet_loss = self.get_gaussian_net_loss()
            return [self.xent_loss, self.gaussian_dnet_loss, self.gaussian_enet_loss]
        '''
        
    ## 1. g_loss
    def _get_cluster_loss(self, Wc):
        cluster_loss = 0
        for i in range(Wc.shape[0]):
            each_row = Wc[i, :]
            for j in range(i + 1, Wc.shape[0]):
                dist_two = torch.pow(Wc[i, :] - Wc[j, :], 2)
                each_loss = torch.relu(self.eta - torch.mean(dist_two)) * self.penalty
                cluster_loss = torch.add(cluster_loss, each_loss)
        out = torch.mean(cluster_loss)
        return out
    
    def get_xent_loss(self):
        cluster_rows = (self.nbin ** 2) * (self.nbin ** 2 - 1) // 2
        cluster_loss = self._get_cluster_loss(self.Wc) / cluster_rows
        
        dna_labels = torch.argmax(self.x_input, dim=-1) # [64, 165]
        dna_logits = self.x_decoded.permute(0,2,1) # [64, 4, 165]
        recon_loss = nn.CrossEntropyLoss()(dna_logits, dna_labels)

        xent_loss = cluster_loss + recon_loss
        return xent_loss
    
    ## 2. d_loss/ e_loss
    def _get_gradient_penalty(self, rand_z, fake_z, mode="z"):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fake_z = fake_z.float()
        rand_z = rand_z.float()
        differences = fake_z - rand_z

        alpha = torch.rand(self.batch_size, 1).to(device) # [64,1], Uniform
        inter_z = (rand_z + (alpha * differences)).requires_grad_(True) ## 必须要requires_grad?
        
        if mode=="z":
            inter_z_score = self.discriminator_z(inter_z)
        elif mode=="y":
            inter_z_score = self.discriminator_y(inter_z)
        else:
            print("The provided mode input is incorrect.\n")
        
        weights = torch.ones_like(inter_z_score).to(device)
        gradients = torch.autograd.grad(outputs=inter_z_score, inputs=inter_z,
                                        grad_outputs=weights,
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]
        
        slopes = torch.sqrt(torch.sum(gradients ** 2, axis=1))
        gradient_penalty = torch.mean((slopes - 1.)**2)
        
        return gradient_penalty
    
    def _get_discriminator_loss(self, rand_z_score, fake_z_score):
        
        loss = -torch.mean(rand_z_score) + torch.mean(fake_z_score)
        return loss
    
    def _get_generator_loss(self, fake_z_score):
        return -torch.mean(fake_z_score)
    
    def get_gaussian_dnet_loss(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        rand_z = torch.randn(self.batch_size, self.z_dim).requires_grad_(True)
        rand_z = rand_z.to(device)
        rand_z_score = self.discriminator_z(rand_z)
        
        fake_z = self.z.detach()
        fake_z_score = self.discriminator_z(fake_z) ## detach
        
        # print( "rand_z_score={}, fake_z_score={}".format(torch.mean(rand_z_score), torch.mean(fake_z_score)) )
        
        GP_z = self._get_gradient_penalty(rand_z, fake_z)
        gaussian_dnet_loss = self._get_discriminator_loss(rand_z_score, fake_z_score) + self.lamda * GP_z
        
        return gaussian_dnet_loss
    
    def get_gaussian_enet_loss(self):
        
        fake_z_score = self.discriminator_z(self.z) ## not detach
        gaussian_enet_loss = self._get_generator_loss(fake_z_score)
        
        return gaussian_enet_loss
    
    '''
    def get_gaussian_net_loss(self):
        rand_z = torch.randn(self.batch_size, self.z_dim).requires_grad_(True)
        rand_z_score = self.discriminator_z(rand_z)
        
        fake_z = self.z.detach()
        # fake_z = self.z
        fake_z_score = self.discriminator_z(fake_z) ## detach

        
        GP_z = self._get_gradient_penalty(rand_z, fake_z)
        gaussian_dnet_loss = self._get_discriminator_loss(rand_z_score, fake_z_score) + self.lamda * GP_z
        
        fake_z_score = self.discriminator_z(self.z) ## not detach
        gaussian_enet_loss = self._get_generator_loss(fake_z_score)
        
        return gaussian_dnet_loss, gaussian_enet_loss
    
    
    def get_expression_net_loss(self):
        # rand: [batch_size, 9]
        # fake: self.softmax_y [batch_size, 9]
        rand_y = torch.randint( self.nbin**2, (self.batch_size,) )
        rand_y = F.one_hot(rand_y, self.nbin**2)
        rand_y_score = self.discriminator_y(rand_y.float())
        fake_y_score = self.discriminator_y(self.softmax_y)
        
        GP_y = self._get_gradient_penalty(rand_y, self.softmax_y, mode="y")
        expression_dnet_loss = self._get_discriminator_loss(rand_y_score, fake_y_score) + self.lamda * GP_y
        expression_enet_loss = self._get_generator_loss( fake_y_score )
        
        return expression_dnet_loss, expression_enet_loss
    
    '''
    ## 3. cls_loss
    
    def get_cls_loss(self):
        # labels: # [64,1]
        logits = self.y # [64,9]
        cls_loss = nn.CrossEntropyLoss()(logits, self.labels)
        return cls_loss
    
    ## 4. others: 
    ## get_cls_accuracy, get_onehot_label, get_predict_label
    
    def get_cls_accuracy(self): # all labels
        logits = self.argmax_y.float()
        labels = self.labels.float()
        num_correct = (labels == logits).float()
        acc = torch.mean(num_correct)
        return acc
        
    
    def get_real_labels(self):
        # self.labels: [64,9]
        labels = self.labels.cpu().detach().numpy()
        labels_EC = [ num // self.nbin for num in labels]
        labels_PA = [ num % self.nbin for num in labels]
        real_labels = [ [label_EC, label_PA] for label_EC,label_PA in zip(labels_EC, labels_PA) ]
        
        return real_labels
    
    def get_predict_logits(self):
        predict_logits = self.softmax_y # [64,9]
        predict_logits = predict_logits.cpu().detach().numpy()
        return predict_logits
        
    
    def get_hamming_distance(self):
        x_onehot = torch.argmax(self.x_input, axis=2) # [batch_size, 165]
        hamming_distance = torch.sum(x_onehot != self.x_recon)/len(self.x_recon)
        return hamming_distance
        
    
    ## sampling for sample_freq and validation
    # self.rep_in, Generator_seq
    def generate_seq(self, batchN=10, z=None, label=None):
        
        if z is None:
            z = torch.randn((self.batch_size*batchN, self.z_dim)) # batchN: batch numbers
        if label is None:
            label = torch.randint(low=0, high=self.nbin ** 2, size=(1,))
        
        label_gen = [label] * self.batch_size
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = z.to(device)
        onehot_label = torch.zeros(self.batch_size, self.nbin**2).to(device)

        for i in range(self.batch_size):
            onehot_label[i, label_gen[i]] = 1  
        
        rep_new = []
        seq_new = []
        for batch_idx in range(batchN):
            cluster_y = torch.matmul(onehot_label, self.Wc)
            x_represented = torch.add( z[ batch_idx*self.batch_size : (batch_idx+1)*self.batch_size, : ], cluster_y) # [batch_size, 64]
            rep_new.extend(x_represented.tolist())
            
            x_decoded = self.decoder(x_represented)  # [batch_size, 165, 4]
            x_softmax = F.softmax(x_decoded, dim=-1) # [batch_size, 165, 4]
            seq_new.extend(onehot2seq(x_softmax.tolist()))
        
        return rep_new, seq_new
    
    ## sampling for validation(must)
    def generate_rep_unsuper(self, onehot):
        x_encoded = self.encoder( onehot.permute(0,2,1) ) 
        z = self.embedding_z(x_encoded)
        # enc = z.tolist()
        
        y = self.embedding_y(x_encoded) 
        argmax_y = torch.argmax(y, dim=-1) 
        softmax_y = F.softmax(y, dim=-1) 
        cluster_y = torch.matmul(softmax_y, self.Wc) # Para for cluster_y, [batch_size, 64]
        
        x_represented = torch.add(z, cluster_y) # output of Represetation, [batch_size, 64]
        enc = x_represented.tolist()
        
        return enc
    
    def generate_exp(self, onehot):
        x_encoded = self.encoder( onehot.permute(0,2,1) ) 
        y = self.embedding_y(x_encoded) 
        argmax_y = torch.argmax(y, dim=-1) 
        return argmax_y.tolist()
    
    ## sampling for validation(optional)
    def generate_z(self, onehot):
        x_encoded = self.encoder( onehot.permute(0,2,1) ) 
        z = self.embedding_z(x_encoded)
        enc = z.tolist()
        
        return enc
    
    def generate_seq_opt(self, sample_number, z=None, label=None, seed=0):
        
        torch.manual_seed(seed)
        if z is None:
            z = torch.randn((sample_number, self.z_dim)) # batchN: batch numbers
        if label is None:
            label = torch.randint(low=0, high=self.nbin ** 2, size=(sample_number,))
        
        label_gen = label
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = z.to(device)
        onehot_label = torch.zeros(sample_number, self.nbin**2).to(device)

        for i in range(sample_number):
            onehot_label[i, label_gen[i]] = 1  
        
        rep_new = []
        seq_new = []
        
        cluster_y = torch.matmul(onehot_label, self.Wc)
        x_represented = torch.add( z, cluster_y)
        rep_new.extend(x_represented.tolist())
        
        x_decoded = self.decoder(x_represented) 
        x_softmax = F.softmax(x_decoded, dim=-1) 
        seq_new.extend(onehot2seq(x_softmax.tolist()))
        
        return rep_new, seq_new
    
    
class AAE_genus:
    def __init__(self, 
                 data_file="../dataset/AAE_finetune/BS_EC_PA_JY_train_comparative_genome_paper_storzdRNAseq.txt", 
                 univ_train_file="../dataset/AAE_univ/univ_train80.txt", 
                 univ_valid_file="../dataset/AAE_univ/univ_test20.txt",
                 supervise_file="../dataset/AAE_supervised/calibration_nm2018.csv", # re-calibration
                 
                 ## AAE_model needed parameters ##
                 nbin = 3,              
                 kernel_size=3, 
                 latent_dim=128, 
                 z_dim=64, 
                 n_layers=8, 
                 batch_size=64, 
                 lamda=10,
                 eta=0.2, 
                 penalty=0.1,
                 savepath = "./check",
                 model_name="aae_genus"
                ):

        self.data, _ = load_seq_data(data_file)
        self.data_train, self.data_valid = split_data(self.data) # (32865, 165, 4) (3652, 165, 4)
        self.data_scale, self.data_seqlen, self.data_dim = self.data_train.shape # 32865 165 4
        ## pretrain mode: train/valid = (1608779, 165, 4) (178754, 165, 4)
        ## finetune mode: train/valid = (32865, 165, 4) (3652, 165, 4)
        
        self.kernel_size, self.latent_dim, self.z_dim, self.n_layers = kernel_size, latent_dim, z_dim, n_layers # 3 128 64 8
        self.nbin, self.batch_size, self.lamda = nbin, batch_size, lamda # 3 64 10
        self.eta, self.penalty = eta, penalty # 0.2 0.1

        _, self.univ_train_seqs = load_seq_data(univ_train_file) # 1706
        _, self.univ_valid_seqs = load_seq_data(univ_valid_file) # 416
        self.supervise_data = load_supervise_data(supervise_file, self.nbin, self.batch_size) 
        # self.supervise_data.next_batch_dict() = (64, 165, 4) (64, ) (64, 165, 4) (64, )
        
        self.savepath, self.model_name = savepath, model_name
        
        self.make_dir()
        self.aae = AAE(self.data_seqlen, self.data_dim, self.nbin, self.kernel_size, self.latent_dim, 
                           self.z_dim, self.n_layers, self.batch_size, self.lamda, self.eta, self.penalty)
        self.apn = APN(self.data_seqlen, self.data_dim, self.nbin, self.kernel_size, self.latent_dim, 
                           self.z_dim, self.n_layers, self.batch_size, self.lamda, self.eta, self.penalty)
        
        self.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
        self.aae = self.aae.to(self.device)
    
    ## creating files for checkpoints, figures, etc.  
    def make_dir(self):
        self.base_dir = os.path.join(self.savepath, self.model_name)
        print("results will be saved in: {}\n".format(self.base_dir))
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if not os.path.exists(self.base_dir + '/figure'):
            os.makedirs(self.base_dir + '/figure')

        if not os.path.exists(self.base_dir + '/training_log'):
            os.makedirs(self.base_dir + '/training_log')

        if not os.path.exists(self.base_dir + '/checkpoints'):
            os.makedirs(self.base_dir + '/checkpoints')
            
        if not os.path.exists(self.base_dir + '/samples'):
            os.makedirs(self.base_dir + '/samples')
    
    ## set learning rate for supervised learning steps
    def set_learning_rate(self,optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    
    ## data generation iterator for unsupervised learning
    def data_iterator(self):
        while True:
            indices = torch.randperm(len(self.data_train))
            for i in range(0, len(indices)-self.batch_size+1, self.batch_size):
                yield self.data_train[indices[i:i+self.batch_size]]    
    
    ## saving plots
    def save_plot(self, filename, data_train, data_valid, control_values):
        
        sns.set_style("darkgrid")
        font = {'size': 12}
        matplotlib.rc('font', **font)
        matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        pdf = PdfPages(self.base_dir + '/figure/' + filename)
        plt.figure()
        plt.plot(np.arange(len(data_train)), data_train)
        plt.plot(np.arange(len(data_valid)), data_valid)
        if control_values is not None:
            plt.plot([0,len(data_train)-1], [control_values[1], control_values[1]])
            plt.legend(['JS_train','JS_valid','JS_control'])
            plt.ylabel('JS Distance')
            plt.xlabel('epoch')
        else:
            plt.legend(['cls_train','cls_valid'])
            plt.ylabel('cls loss(cross entropy)')
            plt.xlabel('supervised epoch')

        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(np.arange(1, len(data_train) + 1))

        pdf.savefig()
        pdf.close()
    
    ### -----------------------------------------------------------------------------------------------------###
    
    def pretrain_ybranch(self,
              learning_rate=1e-4,
              beta1=0.5, 
              beta2=0.9,
              num_epochs=50):
        
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.num_epochs = num_epochs
        
        log_timestamp = time.time()
        log_timestring = datetime.datetime.utcfromtimestamp(log_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        log_writer = SummaryWriter(self.base_dir + '/training_log/APN_' + log_timestring)
        
        optimizer_super = optim.Adam(self.apn.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        
        cls_accuracy_train_list = [] # cls_accuracy_train_all
        cls_accuracy_valid_list = [] # cls_accuracy_valid_all
        
        train_feature = torch.tensor( self.supervise_data.onehot_list_train )
        train_label = torch.tensor( self.supervise_data.label_list_train )
        valid_feature = torch.tensor( self.supervise_data.onehot_list_test )
        valid_label  = torch.tensor( self.supervise_data.label_list_test )
        
        
        train_dataset = SequenceData(train_feature, train_label)
        train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)
        valid_dataset = SequenceData(valid_feature, valid_label)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=self.batch_size, shuffle=True)
        
        counter_train, counter_valid = 0, 0
        for epoch in tqdm(range(1, 1+self.num_epochs)):
            
            self.apn.train()
            for idx,(feature,label) in enumerate(train_dataloader,0):
                outputs = self.apn(feature, label)
                optimizer_super.zero_grad()
                _cls_loss_train = self.apn(feature, label)
                _cls_accuracy_train = self.apn.get_cls_accuracy()
                _cls_loss_train.backward()
                optimizer_super.step()
                
                log_writer.add_scalar("training_super/cls_loss", _cls_loss_train.item(), counter_train)
                log_writer.add_scalar("training_super/cls_accuracy_train", _cls_accuracy_train.item(), counter_train)
                counter_train += 1

            self.apn.eval()
            for idx,(feature,label) in enumerate(valid_dataloader,0):
                _cls_loss_valid = self.apn(feature,label)
                _cls_accuracy_valid = self.apn.get_cls_accuracy()
                log_writer.add_scalar("training_super/cls_accuracy_valid", _cls_accuracy_valid.item(), counter_valid)
                counter_valid += 1

            
            embedding_y_weights = self.apn.embedding_y.state_dict()
            torch.save(self.apn.state_dict(), self.base_dir + '/checkpoints/apn.pth')
            torch.save(embedding_y_weights, self.base_dir + '/checkpoints/embedding_y.pth')
            
    ### -----------------------------------------------------------------------------------------------------###
    
    def train(self,
              learning_rate=1e-4,   
              beta1=0.5, 
              beta2=0.9, 
              sample_freq=150, 
              supervise_freq=50, 
              plot_epoch=20,
              num_epochs=1000,
              save_epoch=50,
              
              frozen_key = None, # emby, full
              frozen_checkpoints = None
              ): 
        
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.iteration = self.data_scale // self.batch_size # 513
        self.sample_freq = min(sample_freq, self.iteration)
        self.supervise_freq = min(supervise_freq, self.iteration)
        self.plot_epoch, self.num_epochs = plot_epoch, num_epochs
        self.save_epoch = save_epoch
        
        ## loading frozen embedding_y layers
        if frozen_checkpoints is not None:
            print("[Frozen] Loading Frozen Checkpoints... \n")
            frozen_model = torch.load(frozen_checkpoints)
            model_dict = self.aae.state_dict()
            state_dict = {k:v for k,v in frozen_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.aae.load_state_dict(model_dict)
            
            if frozen_key == "full":
                for model in [self.aae.encoder, self.aae.embedding_y]:
                    for p in model.parameters():
                        p.requires_grad = False
            elif frozen_key == "emby":
                for p in self.aae.embedding_y.parameters():
                    p.requires_grad = False
        else:
            print("[Direct] Randomly Initialization for Pretrain/Direct Process... \n")
        

        kmer_data_train = [kmer_statistics(i, onehot2seq(self.data_train)) for i in [4,6,8]] # 256 4096 65536
        kmer_data_valid = [kmer_statistics(i, onehot2seq(self.data_valid)) for i in [4,6,8]]
        js_data_train_valid = [kmer_data_train[i].js_with(kmer_data_valid[i]) for i in range(3)]
        print('JS divergence between data_train and data_valid: 4mer: {}, 6mer: {}, 8mer: {}'.format(js_data_train_valid[0],js_data_train_valid[1],js_data_train_valid[2]))
        
        kmer_univ_train = [kmer_statistics(i, self.univ_train_seqs) for i in [4,6,8]]
        kmer_univ_valid = [kmer_statistics(i, self.univ_valid_seqs) for i in [4,6,8]]
        js_univ_train_valid = [kmer_univ_train[i].js_with(kmer_univ_valid[i]) for i in range(3)]
        print('JS divergence between univ_train and univ_valid: 4mer: {}, 6mer: {}, 8mer: {}\n'.format(js_univ_train_valid[0],js_univ_train_valid[1],js_univ_train_valid[2]))
        
        
        log_timestamp = time.time()
        log_timestring = datetime.datetime.utcfromtimestamp(log_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        log_writer = SummaryWriter(self.base_dir + '/training_log/AAE_' + log_timestring) # tensorboard --logdir=C:\Users\dqx18\Desktop\logs\
        
        self.data_train = torch.tensor(self.data_train)
        data_iter = self.data_iterator()
        
        optimizer_super = optim.Adam(self.aae.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        # optimizer_unsuper = optim.Adam(self.aae.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        
        optimizer_xent = optim.Adam(self.aae.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        optimizer_enet = optim.Adam( itertools.chain( self.aae.encoder.parameters(), self.aae.embedding_z.parameters() ) , lr=self.learning_rate, betas=(self.beta1, self.beta2))
        optimizer_dnet = optim.Adam( self.aae.discriminator_z.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        
        counter_iteration = 1 # counter
        counter_earlystopping = 0 # conv
        best_js_fake_data_valid = 1 # best_js
        
        fake_univ_seq = [] # gen_univ_seq
        js_fake_data_train_6mer = [] # train_6js
        js_fake_data_valid_6mer = [] # val_6js
        js_fake_univ_train_6mer = [] # train_6js_univ
        js_fake_univ_valid_6mer = [] # val_6js_univ
        cls_accuracy_train_list = [] # cls_accuracy_train_all
        cls_accuracy_valid_list = [] # cls_accuracy_valid_all
        
        for epoch in range(1, 1+self.num_epochs):
              
            _tqdm = tqdm(range(1, 1+self.iteration))
            for batch_idx in _tqdm:
                _tqdm.set_description('epoch: {}/{}'.format(epoch, self.num_epochs))
                
                _feature = data_iter.__next__() # current data
                _feature = _feature.to(self.device)
                
                # _xent_loss, _gaussian_dnet_loss, _gaussian_enet_loss = self.aae(_feature)
                
                counter_iteration +=1
                
                ### discriminator cost
                for model in [self.aae.discriminator_z, self.aae.discriminator_y]:
                    for p in model.parameters():
                        p.requires_grad = True
                
                _x_recon = self.aae(_feature)
                optimizer_xent.zero_grad()
                _xent_loss = self.aae.get_xent_loss()
                _xent_loss.backward()
                optimizer_xent.step()
                
                _x_recon = self.aae(_feature)
                optimizer_dnet.zero_grad()
                _gaussian_dnet_loss = self.aae.get_gaussian_dnet_loss()
                _gaussian_dnet_loss.backward()
                optimizer_dnet.step()
                
                '''
                print( "xent_loss={}, gaussian_dnet_loss={}".format(_xent_loss, _gaussian_dnet_loss) )
                gaussian_dnet_loss 似乎会随着训练的过程暴涨?
                '''
                
                ### generator cost
                for model in [self.aae.discriminator_z, self.aae.discriminator_y]:
                    for p in model.parameters():
                        p.requires_grad = False
                
                _x_recon = self.aae(_feature)
                optimizer_enet.zero_grad()
                _gaussian_enet_loss = self.aae.get_gaussian_enet_loss()
                _gaussian_enet_loss.backward()
                optimizer_enet.step()
                
                
                _hamming_distance = self.aae.get_hamming_distance()
                
                log_writer.add_scalar("training_unsuper/xent_loss", _xent_loss.item(), counter_iteration)
                log_writer.add_scalar("training_unsuper/gaussian_dnet_loss", _gaussian_dnet_loss.item(), counter_iteration)
                log_writer.add_scalar("training_unsuper/gaussian_enet_loss", _gaussian_enet_loss.item(), counter_iteration)
                log_writer.add_scalar("training_unsuper/hamming_distance", _hamming_distance.item(), counter_iteration)
                
                ### -------------------------------- ###
                ### histogram: monitoring the params ###
                # log_writer.add_scalar("training_values/z_std",  np.std( self.aae.z[:,0].clone().cpu().data.numpy()), counter_iteration)
                # log_writer.add_scalar("training_values/z_mean", np.mean(self.aae.z[:,0].clone().cpu().data.numpy()), counter_iteration)
                
                # for model in [self.aae.discriminator_z, self.aae.embedding_y, self.aae.encoder]:
                #     for name, param in model.named_parameters():
                #         log_writer.add_histogram(tag="training_parameters/" + name, values=param.clone().cpu().data.numpy(), global_step=counter_iteration)
                ### -------------------------------- ###
                
                
                
                if np.mod(batch_idx, self.sample_freq) == 0:
                    generated_rep, generated_seq = self.aae.generate_seq(label=8, batchN=40) # 8: [2,2]
                    generated_rep, generated_seq = np.array(generated_rep), np.array(generated_seq)
                    fake_univ_seq = generated_seq
                    
                    with open(self.base_dir + '/samples/AAE_generated_rep.pkl', 'wb') as f:
                        pickle.dump(generated_rep, f) 
                    write_seq(self.base_dir + "/samples/train_{:02d}_{:05d}.txt".format(epoch, batch_idx), generated_seq)

                ## pretraining mode: unsupervised
                if np.mod(batch_idx, self.supervise_freq) == 0:
                    _supervise_batch = self.supervise_data.next_batch_dict(labeluse=True)
                    _supervise_data_train = _supervise_batch["data_train"] # (64, 165, 4)
                    _supervise_label_train = _supervise_batch["label_train"] # (64, )
                    _supervise_data_valid = _supervise_batch["data_valid"] # (64, 165, 4)
                    _supervise_label_valid = _supervise_batch["label_valid"] # (64, )
                    
                    _supervise_data_train = torch.tensor(_supervise_data_train).to(self.device)
                    _supervise_label_train = torch.tensor(_supervise_label_train).to(self.device)
                    _supervise_data_valid = torch.tensor(_supervise_data_valid).to(self.device)
                    _supervise_label_valid = torch.tensor(_supervise_label_valid).to(self.device)
                    
                    _x_recon = self.aae(_supervise_data_train, _supervise_label_train)
                    _cls_loss_train = self.aae.get_cls_loss()
                    _cls_accuracy_train = self.aae.get_cls_accuracy()
                    
                    _x_recon = self.aae(_supervise_data_valid, _supervise_label_valid)
                    _cls_loss_valid = self.aae.get_cls_loss()
                    _cls_accuracy_valid = self.aae.get_cls_accuracy()
                    
                    optimizer_super.zero_grad()
                    log_writer.add_scalar("training_super/cls_loss", _cls_loss_train.item(), counter_iteration)
                    log_writer.add_scalar("training_super/cls_accuracy_train", _cls_accuracy_train.item(), counter_iteration)
                    log_writer.add_scalar("training_super/cls_accuracy_valid", _cls_accuracy_valid.item(), counter_iteration)
                    
                    ## if frozen whole y branch, annotate this part ##
                    if frozen_key != "full":
                        _cls_loss_train.backward()
                        optimizer_super.step()
                    
                    cls_accuracy_train_list.append(_cls_accuracy_train.item())
                    cls_accuracy_valid_list.append(_cls_accuracy_valid.item())
            
            kmer_fake = [kmer_statistics(i, self.aae.generate_seq()[1]) for i in [4,6,8]]
            js_fake_data_train = [kmer_fake[i].js_with(kmer_data_train[i]) for i in range(3)]
            js_fake_data_valid = [kmer_fake[i].js_with(kmer_data_valid[i]) for i in range(3)]

            kmer_fake_univ = [kmer_statistics(i, fake_univ_seq) for i in [4,6,8]]
            js_fake_univ_train = [kmer_fake_univ[i].js_with(kmer_univ_train[i]) for i in range(3)]
            js_fake_univ_valid = [kmer_fake_univ[i].js_with(kmer_univ_valid[i]) for i in range(3)]

            log_writer.add_scalars("evaluation_kmer/js_fake_data_train", {'4mer': js_fake_data_train[0], '6mer': js_fake_data_train[1], 
                                                                          '8mer': js_fake_data_train[2]}, epoch)
            log_writer.add_scalars("evaluation_kmer/js_fake_data_valid", {'4mer': js_fake_data_valid[0], '6mer': js_fake_data_valid[1], 
                                                                          '8mer': js_fake_data_valid[2]}, epoch)
            log_writer.add_scalars("evaluation_kmer/js_fake_univ_train", {'4mer': js_fake_univ_train[0], '6mer': js_fake_univ_train[1], 
                                                                          '8mer': js_fake_univ_train[2]}, epoch)
            log_writer.add_scalars("evaluation_kmer/js_fake_univ_valid", {'4mer': js_fake_univ_valid[0], '6mer': js_fake_univ_valid[1], 
                                                                          '8mer': js_fake_univ_valid[2]}, epoch)

            js_fake_data_train_6mer.append(js_fake_data_train[1]) # train_6js
            js_fake_data_valid_6mer.append(js_fake_data_valid[1]) # val_6js
            js_fake_univ_train_6mer.append(js_fake_univ_train[1]) # train_6js_univ
            js_fake_univ_valid_6mer.append(js_fake_univ_valid[1]) # val_6js_univ
            
            ### saving checkpoints
            torch.save(self.aae.state_dict(), self.base_dir + '/checkpoints/aae.pth')
            if np.mod(epoch, self.save_epoch) == 0:
                torch.save(self.aae.state_dict(), self.base_dir + '/checkpoints/aae_{}.pth'.format(epoch))
            
            ### plotting evaluations
            if np.mod(epoch, self.plot_epoch) == 0:
                self.save_plot('js_fake_data.pdf', js_fake_data_train_6mer, js_fake_data_valid_6mer, js_data_train_valid)
                self.save_plot('js_fake_univ.pdf', js_fake_univ_train_6mer, js_fake_univ_valid_6mer, js_univ_train_valid)
                self.save_plot('cls_loss_byepoch.pdf', cls_accuracy_train_list, cls_accuracy_valid_list, None)
                
        return

if __name__ == "__main__":
    
    pretrain_file = "../dataset/AAE_pretrain/pretrain_seqs.txt"
    
    data_file = "../dataset/AAE_finetune/BS_EC_PA_JY_train_comparative_genome_paper_storzdRNAseq.txt"
    univ_train_file = "../dataset/AAE_univ/univ_train80.txt"
    univ_valid_file = "../dataset/AAE_univ/univ_test20.txt"
    supervise_file = "../dataset/AAE_supervised/calibration_nm2018.csv"
    
    '''
    Step1: training
    '''
    
    aae_genus = AAE_genus(data_file=data_file, model_name="aae_genus")
    aae_genus.train(num_epochs=600, frozen_checkpoints=None, supervise_freq=1)
    

    '''
    step2: evaluation
    '''

    ori_tag = "aae_genus"
    key_tag = "tags_" + ori_tag

    model_path = "./check/" + ori_tag + "/checkpoints/aae.pth"
    
    tags_evaluation(supervise_file, info_dir = "../results/" + key_tag)
    accs_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag, mode="AAE") # AAE classification acc
    reps_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag) # Fig2a EC, PA
    zspace_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag, dim=0) # zspace dim0
    zspace_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag, dim=1) # zspace dim1
    categorical_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag) # conditional generation
    stack_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag) # classification for each label
    extraploation_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag) # classification for novel
    species_evaluation(model_path, supervise_file, info_dir = "../results/" + key_tag) # Fig2a 13species
    

    
    
    
    
    
    
    