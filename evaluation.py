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



'''
1.GC content and kmer
'''
print(' [*] Cal GC+kmer...')
compare_file_list=['EConly_test20.txt','PAonly_test20.txt','EC_PA_test20.txt',input_file]
def cal_kmer_r_matrix(generated_seq_filename_list,compare_file_list,name='compare',numk=6):
    kmer_r_matrix=[]
    evalu=crosspro.Evaluation.evalu(data_dir=sample_dir,save_dir=evalu_dir)#---------2.dir
    for evalu_file in generated_seq_filename_list: #
        kmer_each_binevaluation_with7=[]
        for compare_file in compare_file_list: 
            print('Cal '+'['+str(name)+']'+compare_file+'__'+evalu_file+'_'+str(numk)+'mer...')
            source1 = './'+compare_file 
            evalu_file=evalu_file.split('.')[0] 
            target = sample_dir 
            shutil.copy(source1, target)
            color_list=['orange','blue','black','brown','darkblue','purple','grey','yellow','green','red','orange','green']
            species_list=[evalu_file,compare_file.strip('.txt')] 
            colorList=['orange'] 
            #1.GC content
            evalu.GC_plot(species_list,color_list)   
            evalu.polyN_plot(species_list,color_list,4,10)
            evalu.kmer_statistic(species_list,numk=numk)
            kmer_r=evalu.kmer_plot(species_list,compare_file.strip('.txt'),'cor',colorList,species_list=species_list,numk=numk) 
            kmer_each_binevaluation_with7.append(kmer_r)
            plt.cla()
            plt.close("all")  
        kmer_r_matrix.append(kmer_each_binevaluation_with7)
    kmer_r_matrix=pd.DataFrame(np.array(kmer_r_matrix))#,columns=['BSonly','EConly','PAonly','BS_EC','BS_PA','EC_PA','univ','all_seq'])#index =generated_bin_list
    kmer_r_matrix.to_csv(evalu_dir+'kmer_cor_bin_situation_'+str(name)+'_'+str(numk)+'mer.csv')
    print(kmer_r_matrix)   
    return 
#2.calculate the kmer correlation matrix
k=6
cal_kmer_r_matrix(generated_seq_filename_list,compare_file_list,name='compare',numk=k)
cal_kmer_r_matrix(compare_file_list,compare_file_list,name='control',numk=k) 
print(' [*] Feature1.1+1.2:GC+kmer done')



'''
2. latent reprensentation
'''
#The latent space
print(' [*] Cal latent space...')
name_list_7=['onlyEC_3_test20_spe2','onlyPA_3_test20_spe2','EC_PA_3_test20_spe2'] 
name_list_3=['EConly_exp_spe2','PAonly_exp_spe2','EC_PA_exp_spe2']
rep_data_7,speflag_7=gen.get_z(rep_dir=rep_dir,name_list_3=name_list_3,name_list_7=name_list_7,gen_bs=256,exp_flag=False)#因为BS_EC只有42条,因此,把batch size 设小一点,不然隐空间不显示这些点
print(' [*] latent rep_data+speflag finished')
latent_rep=crosspro.Represent.latent_rep(rep_data_7=rep_data_7,speflag_7=speflag_7,rep_dir=rep_dir)
X_tsne_7,X_pca_7,X_umap_7=latent_rep.cal_tSNE_PCA(rep_data=rep_data_7,learning_rate=100,perplexity=12)
#denstiy
name_list_7.append('AAE_generated')
latent_rep.plot_kenrel_density_BS_EC_PA(X_tsne_7,'tSNE',name_list_7,200)
latent_rep.plot_kenrel_density_BS_EC_PA(X_pca_7,'PCA',name_list_7,200)
latent_rep.plot_kenrel_density_BS_EC_PA(X_umap_7,'UMAP',name_list_7,200)
#
cluster_method=['AgglomerativeClustering','DBSCAN','OPTICS']
latent_rep.get_quantitative_clustering_result(n_clusters=3,connectmatrix=True,method_list=cluster_method,spe_only3=['onlyEC_3_test20_spe2','onlyBS_3_test20_spe2','onlyPA_3_test20_spe2']) ##'DBSCAN' #'OPTICS'
latent_rep.cluster_plot()  
print(' [*] Feature2:latent space done')


'''
3.Simlarity 
'''
print(' [*] Cal similarity by edit distance...')
def seq2compare(seqs): 
    Length=len(seqs[0])
    map={'A':0,'G':1,'C':2,'T':3}
    seqall=[]
    for seq in seqs:
        eachseq=np.zeros([Length],dtype = 'float')
        for j in range(len(seq)):
            eachseq[j]=map[seq[j]]
        seqall.append(eachseq)
    return np.array(seqall)

def cal_hamming(promoter,AAE_generated_seq):
    hamm_dist_sum=0
    promoter=seq2compare(promoter)
    AAE_generated_seq=seq2compare(AAE_generated_seq)
    hamm_dist=[]
    for i in range(len(promoter)):
        for j in range(len(AAE_generated_seq)):
            NBase_diff=np.sum(promoter[i]!=AAE_generated_seq[j])
            hamm_dist_sum+=NBase_diff
            hamm_dist.append(NBase_diff)
    hamm_dist_mean=hamm_dist_sum/(len(promoter)*len(AAE_generated_seq))
    hamm_dist = np.array(hamm_dist)
    hamm_dist_max=np.max(hamm_dist)
    hamm_dist_min=np.min(hamm_dist)
    return hamm_dist_mean,hamm_dist_max,hamm_dist_min,hamm_dist
    
def plot_hamming_dist(hamm_dist,name):
    pdf = PdfPages(similarity_dir+'/Hamming_distance_'+str(name)+'.pdf')
    plt.figure(figsize=(10,10))
    sns.distplot(hamm_dist,kde=False)
    plt.xlabel('Hamming distance')
    pdf.savefig()
    pdf.close()
    return

hamm_dist_mean,hamm_dist_max,hamm_dist_min,hamm_dist_AI_nature=cal_hamming(gen.inputseqs,generated_seq_list)
print('Similarity: average hamming distance between AI-nature:'+str(hamm_dist_mean)+'bp of 165bp, min: '+str(hamm_dist_min)+',max: '+str(hamm_dist_max))
plot_hamming_dist(hamm_dist_AI_nature,'AI_nature')
#
hamm_dist_mean,hamm_dist_max,hamm_dist_min,hamm_dist_AI_AI=cal_hamming(generated_seq_list,generated_seq_list)
print('Similarity: average hamming distance between AI-AI:'+str(hamm_dist_mean)+'bp of 165bp, min: '+str(hamm_dist_min)+',max: '+str(hamm_dist_max))
plot_hamming_dist(hamm_dist_AI_AI,'AI_AI')
print(' [*] Feature3:similarity(Edit distance) done')