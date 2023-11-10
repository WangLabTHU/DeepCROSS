import Crosspro as crosspro
import os
import sys
import shutil
import numpy as np
import argparse
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from heatmapcluster import heatmapcluster
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=config)
sys.setrecursionlimit(3000) 


log_dir= './output'
generative_model_dir=log_dir+'/generative_model'
sample_dir=log_dir+'/designed_sequences'
evalu_dir_all=log_dir+'/evaluation'
evalu_dir = evalu_dir_all+'/basic_statistic/'
rep_dir=evalu_dir_all+'/latent_representation/'
similarity_dir=log_dir+'/similarity_ratio/'
#
GA_dir=log_dir+'/genetic_algorithm/'
pred_weight_dir_evalu_GA=GA_dir+'/predictor_weight_dir_evalu/'
predictor_trainop_dir=log_dir+'/predictor_trainop'
predictor_evaluop_dir=log_dir+'/predictor_evaluop/'
mkdir=[log_dir,generative_model_dir,sample_dir,evalu_dir_all,evalu_dir,rep_dir,similarity_dir,GA_dir,predictor_trainop_dir,predictor_evaluop_dir]
for d in mkdir:
    if os.path.exists(d) == False:
        os.makedirs(d)

#Input parameters
parser = argparse.ArgumentParser()
parser.description='Please enter three parameters sequence file, sequence-activity paired training file and sequence-activity paired testing file  ...'
parser.add_argument("-seq", "--sequence_file", help="this is the name of sequence_file", type=str)
parser.add_argument("-train", "--sequence-activity(training)", help="this is the name of sequence-activity file(training)",  type=str)
parser.add_argument("-test", "--sequence-activity(testing)", help="this is the name of sequence-activity file(testing)",  type=str)

'''
Generative model
'''
#Input data and generative model:
input_file='finetune_seq.txt'
nbin=3
gen = crosspro.Generators.AAE_semi_maxpooladd_2spe(log_dir=log_dir,nbin=nbin)
gen.BuildModel(input_file,univ_file='univ_train80.txt',univ_val_file='univ_test20.txt',BATCH_SIZE=256,supervise_file='EC_PA_exp_bin_3_train80_originnobin_repeatsample.npy') 
gen.Train(epoch=200,sample_dir=sample_dir,checkpoint_dir=generative_model_dir,log_dir=log_dir)#supervise_freq=10
gen.load(checkpoint_dir = generative_model_dir, model_name='aae')
print(' [*] Generator(AAE) loaded')
#Generate sequences:
generated_seq_filename_list=[]
generated_seq_list=[]
for i in range(0,nbin,1): 
    for j in range(0,nbin,1):  
        for k in range(0,nbin,1):
            print(str(i)+'_'+str(j)+'_'+str(k))
            generated_seq_list.extend(gen.Generate_rep_seq_getanybin(label_gen=[i,j,k]))
            generated_seq_filename_list.append('generated_seq_'+str(i)+'_'+str(j)+'_'+str(k))

#Generate sequences:
generated_bin_list=['2_0','0_2','2_2']
generated_seq_list=[]
generated_seq_filename_list=[]
for bins in generated_bin_list:
    i = bins.split('_')[0]
    j = bins.split('_')[1]
    print(str(i)+'_'+str(j))
    generated_seq_list.extend(gen.Generate_rep_seq_getanybin(label_gen=[i,j],batchN=10))
    generated_seq_filename_list.append('generated_seq_'+bins)

