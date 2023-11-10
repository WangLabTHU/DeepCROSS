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
parser.add_argument("-dataset", "--dataset", help="The sequence and activity dataset in npy format ",  type=str)

#Predictive model
pred= crosspro.Predictors.CNN_crosssignal(train_data='spe3_crosssignal_train70.npy')
pred.BuildModel(DIM=128)
print(' [*] Cross signal predictor training begin..')
pred.Train(weight_dir=predictor_trainop_dir)
pred.load()
pred.predict_val_value()
print(' [*] predictor loaded')  
