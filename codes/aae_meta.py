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


class AAE_meta:
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
                 model_name="aae_meta",
                 mode = "round1"
                ):

        self.data, _ = load_seq_data(data_file)
        self.data_train, self.data_valid = split_data(self.data) # (32865, 165, 4) (3652, 165, 4)
        self.data_scale, self.data_seqlen, self.data_dim = self.data_train.shape # 32865 165 4
        ## pretrain mode: train/valid = (1608779, 165, 4) (178754, 165, 4)
        ## finetune mode: train/valid = (32865, 165, 4) (3652, 165, 4)
        
        self.kernel_size, self.latent_dim, self.z_dim, self.n_layers = kernel_size, latent_dim, z_dim, n_layers # 3 128 64 8
        self.nbin, self.batch_size, self.lamda = nbin, batch_size, lamda # 3 64 10
        self.eta, self.penalty = eta, penalty # 0.2 0.1
        self.savepath, self.model_name, self.mode = savepath, model_name, mode

        _, self.univ_train_seqs = load_seq_data(univ_train_file) # 1706
        _, self.univ_valid_seqs = load_seq_data(univ_valid_file) # 416
        self.supervise_data = load_supervise_data(supervise_file, self.nbin, self.batch_size, self.mode) 
        # self.supervise_data.next_batch_dict() = (64, 165, 4) (64, ) (64, 165, 4) (64, )
        
        
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
    
    def train(self,
              learning_rate=1e-4,   
              beta1=0.5, 
              beta2=0.9, 
              sample_freq=150, 
              supervise_freq=50, 
              plot_epoch=20,
              num_epochs=1000,
              save_epoch=50,

              pretrained_checkpoints = None
              ): 
        
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.iteration = self.data_scale // self.batch_size # 513
        self.sample_freq = min(sample_freq, self.iteration)
        self.supervise_freq = min(supervise_freq, self.iteration)
        self.plot_epoch, self.num_epochs = plot_epoch, num_epochs
        self.save_epoch = save_epoch
        
        if pretrained_checkpoints is not None:
            print("[Pretrain] Loading Pretrained Checkpoints... \n")
            pretrained_model = torch.load(pretrained_checkpoints)
            model_dict = self.aae.state_dict()
            state_dict = {k:v for k,v in pretrained_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.aae.load_state_dict(model_dict)
            
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
    finetune_file = "../dataset/AAE_finetune/BS_EC_PA_JY_train_comparative_genome_paper_storzdRNAseq.txt"
    
    univ_train_file = "../dataset/AAE_univ/univ_train80.txt"
    univ_valid_file = "../dataset/AAE_univ/univ_test20.txt"
    supervise_file = "../dataset/AAE_supervised/calibration_nm2018.csv"
    
    
    '''
    step1: training, pretraining + finetuning
    '''
    
    aae_pretrain = AAE_meta(data_file=pretrain_file, model_name="aae_meta_pretrain")
    aae_pretrain.train(num_epochs=50, pretrained_checkpoints=None, supervise_freq=1, learning_rate=1e-5, save_epoch=1)
    
    pretrained_model = "./check/aae_meta_pretrain/checkpoints/aae_20.pth"
    aae_finetune = AAE_meta(data_file=finetune_file, model_name="aae_meta_finetune")
    aae_finetune.train(num_epochs=600, pretrained_checkpoints=pretrained_model, supervise_freq=1)

    
    '''
    step2: evaluation
    '''
    
    
    ori_tag = "aae_meta_finetune"
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
    
    '''
    step3: round 2, means that after the acquiration of Lib-2, the retraining process of generative models
    '''
    
    round2_finetune_file = "../dataset/AAE_finetune/BS_EC_PA_JY_train_comparative_genome_paper_storzdRNAseq.txt"
    round2_univ_train_file = "../dataset/AAE_univ/round2_univ_train80.txt"
    round2_univ_valid_file = "../dataset/AAE_univ/round2_univ_test20.txt"
    round2_supervise_file = "../dataset/AAE_supervised/calibration_mpra2024.csv" 
    
    pretrained_model = "./check/aae_meta_finetune/checkpoints/aae.pth"
    aae_round2 = AAE_meta(data_file=round2_finetune_file, model_name="aae_meta_round2", mode="round2",
                          univ_train_file=round2_univ_train_file,
                          univ_valid_file=round2_univ_valid_file,
                          supervise_file=round2_supervise_file)
    aae_round2.train(num_epochs=600, pretrained_checkpoints=pretrained_model, supervise_freq=1)
    
    ###-----------------------------------------------------------------------------------------###
    ###-----------------------------------------------------------------------------------------###
    
    ori_tag = "aae_meta_round2"
    key_tag = "tags_" + ori_tag
    model_path = "./check/" + ori_tag + "/checkpoints/aae.pth"
    tags_evaluation(round2_supervise_file, info_dir = "../results/" + key_tag, round_tag="round2")
    accs_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, mode="AAE", round_tag="round2")
    reps_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, round_tag="round2") 
    zspace_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, dim=0, round_tag="round2") 
    zspace_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, dim=1, round_tag="round2") 
    categorical_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, round_tag="round2") 
    stack_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag, round_tag="round2") 
    extraploation_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag) 
    species_evaluation(model_path, round2_supervise_file, info_dir = "../results/" + key_tag) 