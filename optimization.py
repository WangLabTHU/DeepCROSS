import DeepCROSS as crosspro
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
parser.add_argument("-save_freq_GA", "--iteration of sequence saving", help="iteration of sequence saving", type=int,default="10")
parser.add_argument("-MaxIter", "--Max iterations ", help="Max number of iterations in the genetic algorithm",  type=int,default="5000")


##optimization
scales=['ALL','Small_test','final']
#running scale
scale=scales[2]
print(' [*] Running scale: '+str(scale))
if scale == 'all':
    save_freq=100
    Nrange=50 
    Nrangemin=50 
    MaxIter=5000
elif scale == 'Small_test':
    save_freq=10 
    Nrange=4
    Nrangemin=3
    MaxIter=5000
else:
    save_freq_GA=10 #
    save_freq_GD=2
    Nrange=50 
    Nrangemin=2 
    MaxIter=5000
print('save_freq_GA: '+str(save_freq_GA))
print('save_freq_GD: '+str(save_freq_GD))
print('Nrange: '+str(Nrange))
'''
遗传算法优化
'''
print(' [*] GA Optimizing seq ...')
GA = crosspro.Optimizers.GeneticAthm(gen.Generator,pred.Predictor)
Nrangemin_GA = GA.run(outdir=GA_dir,save_freq=save_freq_GA,MaxIter=MaxIter) 
print(' [*] Nrangemin_GA: '+str(Nrangemin_GA))
'''
Evaluation: the predicted activity after each iteration
'''                        
def evalu_boxplot_linplot(result_dir,method,Nrange=Nrange,save_freq=save_freq_GA):
    #1.boxplot
    exp_GA_total=[]
    flist=[str((i+1)*save_freq) for i in range(Nrange)] 
    for fn in flist:
        exp_GA_total.append(np.load(result_dir+'ExpIter'+fn+'.npy')) 
    exp_GA_total=np.array(exp_GA_total).transpose()
    print('exp_GA_total:') 
    print(exp_GA_total.shape)
    plt.figure(figsize=(30, 10))
    pdf = PdfPages(result_dir+'/'+method+'_eachepoch_boxplot.pdf')
    plt.boxplot(exp_GA_total,showfliers=False)
    plt.grid(linestyle='--')
    pdf.savefig() 
    pdf.close()
    print(' [*] '+method+'_Polt1_boxplot Done!')
    #2.lineplot
    exp_GA_total=[]
    flist=[str((i+1)*save_freq) for i in range(Nrange)] 
    for fn in flist:
        exp_GA_total.append(np.load(result_dir+'ExpIter'+fn+'.npy')) 
    exp_GA_total=np.array(exp_GA_total).transpose() #-----2.
    minexp=np.min(exp_GA_total,0)
    maxexp=np.max(exp_GA_total,0)
    min5=np.percentile(exp_GA_total, 5,axis=0)
    max95=np.percentile(exp_GA_total, 95,axis=0)
    medianexp=np.median(exp_GA_total,0)
    plt.figure(figsize=(20, 10))
    pdf = PdfPages(result_dir+'/'+method+'_eachepoch_line.pdf')
    plt.plot(range(Nrange),medianexp,linewidth=2.5,color='black')
    plt.fill_between(range(Nrange), min5, max95, facecolor='blue', alpha=0.5)
    plt.fill_between(range(Nrange), minexp, maxexp, facecolor='blue', alpha=0.3)
    pdf.savefig() 
    pdf.close()
    print(' [*] '+method+'_Polt2_line Done!')
    return

evalu_boxplot_linplot(result_dir=GA_dir,method='GA',Nrange=Nrangemin_GA,save_freq=save_freq_GA) 
