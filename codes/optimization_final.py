import os, sys
import numpy as np
import pandas as pd
import math, time
import pickle
from collections import Counter

import umap
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering,DBSCAN,OPTICS, cluster_optics_dbscan
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.neighbors import kneighbors_graph

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

from utils.data import load_seq_data, load_supervise_data, split_data
from aae_genus import *

import re, random
import Levenshtein

EPS = 2 ** (-5)

class GeneticAthm():
    def __init__(self,
                 generator_path,
                 predictor_ec_path,
                 predictor_pa_path,
                 z_dim=64,
                 mode="econly"):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = z_dim
        self.x = np.random.normal(size=(3200,z_dim))
        self.mode = mode
        
        self.Generator = AAE().to(device)
        self.Generator.load_state_dict(torch.load(generator_path))
        self.Generator.eval()
        _, self.Seqs = self.Generator.generate_seq_opt(sample_number=3200, z=torch.tensor(self.x, dtype=torch.float32))
        
        self.Predictor_EC = PredictorNet_language(model_name = "pred_ec", length=165)
        self.Predictor_PA = PredictorNet_language(model_name = "pred_pa", length=165)
        self.predictor_ec_path, self.predictor_pa_path = predictor_ec_path, predictor_pa_path
        
        self.Score = self.score_calculation(self.Seqs)
        self.oh = seq2onehot(self.Seqs, 165)
    
    def score_calculation(self, seqs):

        pred_ec = self.Predictor_EC.predict_input(model_path=self.predictor_ec_path, inputs=seqs, mode="data")
        pred_pa = self.Predictor_PA.predict_input(model_path=self.predictor_pa_path, inputs=seqs, mode="data")
        
        pred_ec, pred_pa = np.array(pred_ec), np.array(pred_pa)
        
        if self.mode == "econly":
            scores = np.where(pred_ec > 3, -np.inf, np.log2(2**pred_ec + EPS) - np.log2(2**pred_pa + EPS))
            # scores = np.log2( 2 ** pred_ec + EPS) - np.log2( 2 ** pred_pa + EPS)
        elif self.mode == "paonly":
            scores = np.where(pred_pa > 5, -np.inf, np.log2(2**pred_pa + EPS) - np.log2(2**pred_ec + EPS))
            # scores = np.log2( 2 ** pred_pa + EPS) - np.log2( 2 ** pred_ec + EPS)
        elif self.mode == "ecpa":
            # scores = np.where(pred_ec > 3, -np.inf, np.where(pred_pa > 5, -np.inf, np.log2(2**pred_ec + EPS) + np.log2(2**pred_pa + EPS)))
            scores = np.log2( 2 ** pred_pa + EPS) + np.log2( 2 ** pred_ec + EPS)
        else:
            print("Error input Modes!")
            return
        return scores
            
        
    def run(self,
            outdir='../results/GAfinal',
            MaxPoolsize=2000,
            P_rep=0.3,
            P_new=0.25,
            P_elite=0.25,
            MaxIter=2000):
        
        self.outdir = outdir
        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)
        self.MaxPoolsize = MaxPoolsize
        self.P_rep = P_rep
        self.P_new = P_new
        self.P_elite = P_elite
        self.MaxIter = MaxIter
        I = np.argsort(self.Score)
        I = I[::-1]
        self.Score = self.Score[I]
        self.x = self.x[I,:]
        self.oh = self.oh[I,:,:]
        scores = []
        convIter = 0
        
        bestscore = np.max(self.score_calculation(self.Seqs))
        
        for iteration in range(1,1+self.MaxIter): 
            Poolsize = self.Score.shape[0]
            Nnew = math.ceil(Poolsize*self.P_new)
            Nelite = math.ceil(Poolsize*self.P_elite)
            IParent = self.select_parent( Nnew, Nelite, Poolsize) 
            Parent = self.x[IParent,:].copy()
            x_new = self.act(Parent) # (800, 64)
            
            _, Seqs_new = self.Generator.generate_seq_opt(sample_number=len(x_new), z=torch.tensor(x_new, dtype=torch.float32))  
            oh_new = seq2onehot(Seqs_new, 165)
            Score_new = self.score_calculation(Seqs_new)
            
            self.x = np.concatenate([self.x, x_new])
            self.oh = np.concatenate([self.oh, oh_new])
            self.Score = np.append(self.Score,Score_new)
            
            I = np.argsort(self.Score)
            I = I[::-1]
            self.x = self.x[I,:]
            self.oh = self.oh[I,:,:]
            self.Score = self.Score[I]
            
            # print(len(set(onehot2seq(self.oh))))
            
            I = self.delRep(self.oh ,P_rep)
            
            self.x = np.delete(self.x,I,axis=0)
            self.oh  = np.delete(self.oh ,I,axis=0)
            self.Score = np.delete(self.Score,I,axis=0)
            
            self.x = self.x[:MaxPoolsize, :]
            self.oh = self.oh[:MaxPoolsize, :, :]
            self.Score = self.Score[:MaxPoolsize]
            
            print('Iter = ' + str(iteration) + ' , BestScore = ' + str(self.Score[0]))
            
            
            if iteration%100 == 0:
                np.save(outdir+'/ExpIter'+str(iteration),self.Score)
                Seq = onehot2seq(self.oh)
                np.save(outdir+'/latentv'+str(iteration),self.x)
                write_seq(outdir+'/SeqIter'+str(iteration)+'.fa',Seq)
                print('Iter {} was saved!'.format(iteration))

                scores.append(self.Score)
                
                if np.max(scores[-1]) > bestscore:
                    bestscore = np.max(scores[-1])
                # else:
                #     break
        pdf = PdfPages(outdir+'/each_iter_distribution.pdf')
        plt.figure()
        plt.boxplot(scores)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        pdf.savefig()
        pdf.close()
        
        pdf = PdfPages(outdir+'/compared_with_natural.pdf')
        plt.figure()
        
        nat_score = self.score_calculation(self.Seqs)
        
            
        plt.boxplot([nat_score,scores[-1]])
        plt.ylabel('Score')
        plt.xticks([1,2],['Natural','Optimized'])
        pdf.savefig()
        pdf.close()
        return
    
    def PMutate(self, z): 
        p = np.random.randint(0,z.shape[0])
        z[p] = np.random.normal()
        return
    
    def Reorganize(self, z, Parent): 
        index = np.random.randint(0, 1,size=(z.shape[0]))
        j = np.random.randint(0, Parent.shape[0])
        for i in range(z.shape[0]):
            if index[i] == 1:
                z[i] = Parent[j,i].copy()
        return
    
    def select_parent(self,Nnew, Nelite, Poolsize):
        ParentFromElite = min(Nelite,Nnew//2)
        ParentFromNormal = min(Poolsize-Nelite, Nnew-ParentFromElite)
        I_e = random.sample([ i for i in range(Nelite)], ParentFromElite)
        I_n = random.sample([ i+Nelite for i in range(Poolsize - Nelite)], ParentFromNormal)
        I = I_e + I_n
        return I

    def act(self, Parent):
        for i in range(Parent.shape[0]):
            action = np.random.randint(0,1)
            if action == 0:
                self.PMutate(Parent[i,:])
            elif action == 1:
                self.Reorganize(Parent[i,:], Parent)
        return Parent
    
    def delRep(self, Onehot, p):
        I = set()
        n = Onehot.shape[0]
        i = 0
        while i < n-1:
            if i not in I:
                a = Onehot[i,:,:]
                a = np.reshape(a,((1,)+a.shape))
                a = np.repeat(a,n-i-1,axis=0)
                I_new = np.where(( np.sum(np.abs(a - Onehot[(i+1):,:,:]),axis=(1,2)) / (Onehot.shape[1]*2) ) < p)[0]
                I_new = I_new + i+1
                I = I|set(I_new)
            i += 1
        return list(I)


def plot_boxplot(plot_path, barplot_data):
    
    sns.set_style("darkgrid")
    font = {'size' : 8}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (20,10), dpi = 600) # (7,11)

    ax = sns.boxplot( x="label", y="prediction",  data=barplot_data,  boxprops=dict(alpha=.9), hue="species", 
                      fliersize=1, flierprops={"marker": 'x'}, palette=["cornflowerblue", "indianred"])
    h,_ = ax.get_legend_handles_labels()

    plt.xticks(rotation=270)
    
    ax.set_xlabel('Controlling', fontsize=10)
    ax.set_ylabel('Predictions', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.tight_layout()
    plt.show()
    plt.savefig(plot_path)


def inner_levenshtein_distance_calculation(samples):
    samples_length = len(samples)    
    res_list = []
    for i in range(samples_length):
        seq_dist = []
        for j in range(samples_length):
            if j!=i:
                dis = Levenshtein.distance(samples[i],samples[j])
                seq_dist.append(dis)
        res_list.append(min(seq_dist))
    avg_dis = np.sum(res_list)/len(res_list)
    return avg_dis

def plot_barplot(plot_path, barplot_data):
    
    sns.set(style='darkgrid')
    font = {'size' : 8}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (20,10), dpi = 600)

    sns.barplot(x="tags", y="distances", data=barplot_data, color= "cornflowerblue", # "#F9A319",
                order = ["core_motifs", "random", "johns_pred_ec2pa2", "genus_pred_ec2pa2", "meta_pred_ec2pa2", "meta_pred_ec1pa0", "meta_pred_ec0pa1",
                         "econly", "paonly", "ecpa", "bothno", "J23119_randflanking", "GA_econly", "GA_paonly", "GA_ecpa"])
    
    plt.xticks(rotation=270)
    
    ax.set_xlabel("Controlling", fontsize=10)
    ax.set_ylabel("Levenshtein Distances", fontsize=10)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.tight_layout()
    plt.show()
    plt.savefig(plot_path)

if __name__ == "__main__":
    
    '''
    step1: sampling through genetic algorithms
    '''
    
    generator_path = "./check/aae_meta_final/checkpoints/aae.pth"
    predictor_ec_path = "./check/pred_ec_final/checkpoint.pth"
    predictor_pa_path = "./check/pred_pa_final/checkpoint.pth"
    
    for tag in ["econly", "paonly", "ecpa"]: # , 
    
        GA_optimizer = GeneticAthm(generator_path = generator_path, predictor_ec_path = predictor_ec_path, predictor_pa_path = predictor_pa_path, mode=tag)
        GA_optimizer.run(outdir='../results/GAfinal/{}/'.format(tag))
    
    '''
    step2: perparing for small scale MPRA library
    '''
    
    seqs_list = []
    pred_list_ec, pred_list_pa = [], []
    scores_list, tags_list = [], []
    
    model_ec = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec_path = "./check/pred_ec_final/checkpoint.pth"
    model_pa = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa_path = "./check/pred_pa_final/checkpoint.pth"
    
    topK = 200
    iters = 2000
    
    seqs_GA_econly = open_fa("../results/GAfinal/econly/SeqIter{}.fa".format(iters))
    pred_GA_econly_ec = model_ec.predict_input(model_path=model_ec_path, inputs=seqs_GA_econly, mode="data")
    pred_GA_econly_pa = model_pa.predict_input(model_path=model_pa_path, inputs=seqs_GA_econly, mode="data")
    seqs_GA_econly, pred_GA_econly_ec, pred_GA_econly_pa = np.array(seqs_GA_econly), np.array(pred_GA_econly_ec), np.array(pred_GA_econly_pa)
    scores = np.log2(2 ** pred_GA_econly_ec + EPS) - np.log2(2 ** pred_GA_econly_pa + EPS)
    seqs_list += list(seqs_GA_econly)[0:topK]
    pred_list_ec += list(pred_GA_econly_ec)[0:topK]
    pred_list_pa += list(pred_GA_econly_pa)[0:topK]
    scores_list += list(scores)[0:topK]
    tags_list += ["GA_econly"] * topK
    
    seqs_GA_paonly = open_fa("../results/GAfinal/paonly/SeqIter{}.fa".format(iters))
    pred_GA_paonly_ec = model_ec.predict_input(model_path=model_ec_path, inputs=seqs_GA_paonly, mode="data")
    pred_GA_paonly_pa = model_pa.predict_input(model_path=model_pa_path, inputs=seqs_GA_paonly, mode="data")
    seqs_GA_paonly, pred_GA_paonly_ec, pred_GA_paonly_pa = np.array(seqs_GA_paonly), np.array(pred_GA_paonly_ec), np.array(pred_GA_paonly_pa)
    scores = np.log2(2 ** pred_GA_paonly_pa + EPS) - np.log2(2 ** pred_GA_paonly_ec + EPS)
    seqs_list += list(seqs_GA_paonly)[0:topK]
    pred_list_ec += list(pred_GA_paonly_ec)[0:topK]
    pred_list_pa += list(pred_GA_paonly_pa)[0:topK]
    scores_list += list(scores)[0:topK]
    tags_list += ["GA_paonly"] * topK
    
    seqs_GA_ecpa = open_fa("../results/GAfinal/ecpa/SeqIter{}.fa".format(iters))
    pred_GA_ecpa_ec = model_ec.predict_input(model_path=model_ec_path, inputs=seqs_GA_ecpa, mode="data")
    pred_GA_ecpa_pa = model_pa.predict_input(model_path=model_pa_path, inputs=seqs_GA_ecpa, mode="data")
    seqs_GA_ecpa, pred_GA_ecpa_ec, pred_GA_ecpa_pa = np.array(seqs_GA_ecpa), np.array(pred_GA_ecpa_ec), np.array(pred_GA_ecpa_pa)
    scores = np.log2(2 ** pred_GA_ecpa_ec + EPS) + np.log2(2 ** pred_GA_ecpa_pa + EPS)
    seqs_list += list(seqs_GA_ecpa)[0:topK]
    pred_list_ec += list(pred_GA_ecpa_ec)[0:topK]
    pred_list_pa += list(pred_GA_ecpa_pa)[0:topK]
    scores_list += list(scores)[0:topK]
    tags_list += ["GA_ecpa"] * topK
    
    df_GA = pd.DataFrame({"seqs": seqs_list, "pred_ec": pred_list_ec, "pred_pa": pred_list_pa, "scores": scores_list, "tags": tags_list})
    df_GA.to_csv("../results/GAfinal/GA_pred_select.csv")
    
    '''
    step3: validation by predictive models
    '''
    
    model_ec = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec_path = "./check/pred_ec_final/checkpoint.pth"
    model_pa = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa_path = "./check/pred_pa_final/checkpoint.pth"
    
    df_mpra = pd.read_csv("../dataset/AAE_supervised/calibration_mpra_20240713.csv")
    rows_to_keep = df_mpra['tags'].isin(["johns_ec2pa2", "genus_ec2pa2", "meta_ec2pa2", "meta_ec1pa0", "meta_ec0pa1"])
    df_mpra = df_mpra[~rows_to_keep]
    seqs_mpra, tags_mpra = list(df_mpra.loc[:,"seqs"]), list(df_mpra.loc[:,"tags"])
    
    df_GA = pd.read_csv("../results/GAfinal/GA_pred_select.csv")
    seqs_GA, tags_GA = list(df_GA.loc[:,"seqs"]), list(df_GA.loc[:,"tags"])
    
    seqs_list = seqs_mpra + seqs_GA
    tags_list = tags_mpra + tags_GA
    
    pred_list_ec = model_ec.predict_input(model_path=model_ec_path, inputs=seqs_list, mode="data")
    pred_list_pa = model_pa.predict_input(model_path=model_pa_path, inputs=seqs_list, mode="data")
    
    pred_list = pred_list_ec + pred_list_pa
    label_list = tags_list + tags_list
    spe_list = ["EC"] *len(tags_list) + ["PA"] * len(tags_list)

    df_valid = pd.DataFrame({"prediction": pred_list, "label": label_list, "species": spe_list})
    
    plot_boxplot("../results/GAfinal/GA_MPRA_comparison.png", df_valid)
    

    '''
    step4: calculation of inner Levenshtein distances
    '''
    
    df_GA = pd.read_csv("../results/GAfinal/GA_pred_select.csv")
    df_mpra = pd.read_csv("../dataset/AAE_supervised/calibration_mpra_20240713.csv")
    rows_to_keep = df_mpra['tags'].isin(["johns_ec2pa2", "genus_ec2pa2", "meta_ec2pa2", "meta_ec1pa0", "meta_ec0pa1"])
    df_mpra = df_mpra[~rows_to_keep]
    seqs_GA, tags_GA = list(df_GA.loc[:,"seqs"]), list(df_GA.loc[:,"tags"])
    seqs_mpra, tags_mpra = list(df_mpra.loc[:,"seqs"]), list(df_mpra.loc[:,"tags"])
    df_valid = pd.DataFrame({"seqs": seqs_mpra + seqs_GA, "tags": tags_mpra + tags_GA})
    
    grouped = df_valid.groupby('tags')
    sub_lists = {tag: grouped.get_group(tag) for tag in grouped.groups}
    dist_list, tag_list = [], []
    
    for tag, sub_df in sub_lists.items():
        seqs, tags = list(sub_df.loc[:,"seqs"]), list(sub_df.loc[:,"tags"])
        seqs, tags = seqs[0:topK], tags[0:topK]
        avg_dis = inner_levenshtein_distance_calculation(seqs)
        dist_list.append(avg_dis)
        tag_list.append(tag)
    
    df_bar = pd.DataFrame({"distances": dist_list, "tags": tag_list})
    plot_barplot("../results/GAfinal/GA_distance.png", df_bar)
    