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


def cal_tSNE_PCA(rep_data=None,learning_rate=100,perplexity=12):
    pca=PCA(n_components=2)
    umaps = umap.UMAP(random_state=21,n_components=2)
    X_tsne = TSNE(n_components=2,learning_rate=100,perplexity=12).fit_transform(rep_data) 
    X_pca = pca.fit_transform(rep_data)
    X_umap = umaps.fit_transform(rep_data)
    return X_tsne,X_pca,X_umap


def plot_kernel_density(savepath, archetypes_TSNE, label_list, order_list, palette="viridis_r", mode="scatter"):
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (8,6), dpi = 600)
    ax.set_xticks([])
    ax.set_yticks([])

    df_concat = pd.DataFrame({"t-sne1": archetypes_TSNE[:,0], "t-sne2": archetypes_TSNE[:,1], "label": label_list})
    order =  order_list

    sns.set_style("darkgrid")
    
    if mode=="scatter":
        g = sns.jointplot(data=df_concat,x='t-sne1',y='t-sne2', hue='label',alpha=0.8, legend=False,
                      hue_order=order, palette=palette) 
        g.plot_joint(sns.scatterplot)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")
    elif mode=="kde":
        sns.kdeplot(data=df_concat,x='t-sne1',y='t-sne2', hue='label', hue_order=order, palette=palette,alpha=0.8, thresh=0.5)
    
    plt.title("")
    plt.show()
    plt.savefig(savepath)

def plot_species(savepath, archetypes_TSNE, label_list, order_list, palette="viridis_r", mode="scatter"):
    
    sns.set_style("darkgrid")
    font = {'size' : 12}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (8,12), dpi = 900)
    ax.set_xticks([])
    ax.set_yticks([])

    df_concat = pd.DataFrame({"t-sne1": archetypes_TSNE[:,0], "t-sne2": archetypes_TSNE[:,1], "label": label_list})
    order =  order_list

    if mode=="scatter":
        
        ax = sns.scatterplot(data=df_concat,x='t-sne1',y='t-sne2', hue='label', hue_order=order, palette=palette, 
                        alpha=1, s=5, linewidth=0) # , marker="."
        handles, labels = ax.get_legend_handles_labels()
        
    elif mode=="kde":
        sns.kdeplot(data=df_concat,x='t-sne1',y='t-sne2', hue='label', hue_order=order, palette=palette,alpha=0.8, thresh=0.5)
    
    ax.set_xlabel('PCA-1', fontsize=10)
    ax.set_ylabel('PCA-2', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.1, 1), ncol=3)
    plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.1, 1), ncol=3)
    
    plt.title("")
    plt.show()
    plt.savefig(savepath)


def plot_barplot(plot_path, barplot_data):
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 600)

    sns.barplot(x="keys", y="values", data=barplot_data, alpha = 0.8, dodge=False)
    ax.set_xlabel("")
    ax.set_ylabel("Values")

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.title("")
    plt.show()
    plt.savefig(plot_path)

def plot_boxplot(plot_path, barplot_data):
    font = {'size' : 8}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 600)

    ax = sns.boxplot( x="label", y="prediction",  data=barplot_data,  boxprops=dict(alpha=.9), hue="species", # hue_order = hue_order,
                      fliersize=1, flierprops={"marker": 'x'}, palette=["cornflowerblue", "indianred"]) # # palette="viridis_r"
    h,_ = ax.get_legend_handles_labels()

    # plt.xticks(rotation=270)
    
    ax.set_xlabel('Controlling', fontsize=10)
    ax.set_ylabel('Predictions', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.show()
    plt.savefig(plot_path)

'''
Function1: evaluate the distribution of semi-supervised training dataset
input: supervised data file
output: distribution plot for each label
'''

def tags_evaluation(supervise_file, info_dir=None, round_tag="round1"):
    
    supervise_data = load_supervise_data(supervise_file, 3, 64, round_tag)
    seqs_list = supervise_data.seqs_list
    label_list = supervise_data.label_list # (9227, )
    binlist = supervise_data.binlist
    expression_list = supervise_data.expression_list
    nbin = supervise_data.nbin
    
    min_ec, max_ec = min(expression_list[:,0]), max(expression_list[:,0])
    min_pa, max_pa = min(expression_list[:,1]), max(expression_list[:,1])
    print("Current binlist for separation: ", binlist)
    print("Minimum/Maximum EC value is: {},{} ".format(min_ec, max_ec))
    print("Minimum/Maximum PA value is: {},{} ".format(min_pa, max_pa))
    
    if info_dir is not None:
        df = pd.DataFrame({"seqs": seqs_list, "label_ec": expression_list[:,0], "label_pa": expression_list[:,1]})
        df.to_csv( os.path.join(info_dir, "round1_data.csv") )
        info = {"bin": binlist, "min_ec": min_ec, "max_ec": max_ec, "min_pa": min_pa, "max_pa": max_pa}
        np.save(os.path.join(info_dir, "round1_info.npy"), info)

        labels_EC = [ num // nbin for num in label_list]
        labels_PA = [ num %  nbin for num in label_list]
        label_list = [ [label_EC, label_PA] for label_EC,label_PA in zip(labels_EC, labels_PA) ]
        
        counter = Counter(tuple(item) for item in label_list)
        counter = dict(counter)
        keys_list, values_list = list(counter.keys()), list(counter.values())
        
        
        idx = sorted(range(len(values_list)), key=lambda k: values_list[k], reverse=True)
        keys_list = np.array(keys_list)[idx]
        values_list = np.array(values_list)[idx]
        keys_list = [str(item) for item in keys_list]
        
        df_bars = pd.DataFrame({"keys": keys_list, "values": values_list})
        plot_barplot(os.path.join(info_dir, "round1_label_barplot.png"), df_bars)

'''
Function2: evaluate the classification accuracy of each label in the trained AAE
input: supervised data file
output: classification accuracy for each label
'''

def accs_evaluation(model_path, supervise_file, info_dir=None, mode="AAE", round_tag="round1"):
    
    if mode == "AAE":
        model = AAE()
    elif mode == "APN":
        model = APN()
    model.load_state_dict(torch.load( model_path ))
    model.eval()
    
    supervise_data = load_supervise_data(supervise_file, 3, 64, round_tag)
    seqs_list = supervise_data.seqs_list
    label_list = supervise_data.label_list
    nbin = supervise_data.nbin
    
    onehot = seq2onehot(seqs_list, 165)
    onehot = torch.tensor(onehot)
    pred_list = model.generate_exp(onehot)
    
    pred_ec = [ num // nbin for num in pred_list]
    pred_pa = [ num % nbin for num in pred_list]  
    label_ec = [ num // nbin for num in label_list]
    label_pa = [ num % nbin for num in label_list]
    
    pred_list = np.array([ [item1, item2] for item1, item2 in zip(pred_ec, pred_pa) ])
    label_list = np.array([ [item1, item2] for item1, item2 in zip(label_ec, label_pa) ])
    
    counter = Counter(tuple(item) for item in label_list)
    counter = dict(counter)
    keys_list, values_list = list(counter.keys()), list(counter.values())
    
    idx = sorted(range(len(values_list)), key=lambda k: values_list[k], reverse=True)
    keys_list = np.array(keys_list)[idx]
    acc_list = []
    
    for label in keys_list:
        mask = (label_list == label).all(axis=1)
        pred_sublist = pred_list[mask]
        label_sublist = label_list[mask]
        num_correct = (label_sublist == pred_sublist).all(axis=1).sum()
        acc = num_correct / len(label_sublist)
        acc_list.append(acc)
    
    print("Accucary: ", acc_list)
    keys_list = [str(item) for item in keys_list]
    df_bars = pd.DataFrame({"keys": keys_list, "values": acc_list})
    plot_barplot(os.path.join(info_dir, "round1_acc_barplot.png"), df_bars)


'''
Function3: plot the representation of data for each class
input: supervised data file
output: representation of each class

mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (1, 1): 4,
        (1, 2): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8
    }
'''

def reps_evaluation(model_path, supervise_file, info_dir=None, round_tag="round1"):
    model = AAE()
    model.load_state_dict(torch.load( model_path ))
    model.eval()
    
    supervise_data = load_supervise_data(supervise_file, 3, 64, round_tag)
    seqs_list = supervise_data.seqs_list
    label_list = supervise_data.label_list
    nbin = supervise_data.nbin
    
    ## selecting the 0,1,3,8 labels for pca
    filtered_seqs_list = []
    filtered_label_list = []
    target_values = [0, 1, 3, 8]
    counters = {label: 0 for label in target_values}
    for label, seqs in zip(label_list, seqs_list):
        if label in target_values and counters[label] < 100:
            filtered_seqs_list.append(seqs)
            filtered_label_list.append(label)
            counters[label] += 1
    seqs_list = filtered_seqs_list
    label_list = filtered_label_list
    
    onehot = seq2onehot(seqs_list, 165)
    onehot = torch.tensor(onehot)
    rep_list = model.generate_rep_unsuper(onehot)
    
    X_tsne,X_pca,X_umap = cal_tSNE_PCA(rep_data=rep_list,learning_rate=100,perplexity=12)
    tag_list = label_list
    name_list = list(set(tag_list))
    
    plot_kernel_density(os.path.join(info_dir, "round1_emb_tsne.png"), X_tsne, tag_list, name_list)
    plot_kernel_density(os.path.join(info_dir, "round1_emb_pca.png"), X_pca, tag_list, name_list)
    plot_kernel_density(os.path.join(info_dir, "round1_emb_umap.png"), X_umap, tag_list, name_list)
    
'''
Function4: evaluate the zspace embedding, to make sure the z ~ N(0,1)
input: supervised data file
output: zspace representation for specified dimension
'''

def zspace_evaluation(model_path, supervise_file, info_dir=None, dim=0, round_tag="round1"):
    model = AAE()
    model.load_state_dict(torch.load( model_path ))
    model.eval()
    
    supervise_data = load_supervise_data(supervise_file, 3, 64, round_tag)
    seqs_list = supervise_data.seqs_list
    label_list = supervise_data.label_list
    nbin = supervise_data.nbin
    
    onehot = seq2onehot(seqs_list, 165)
    onehot = torch.tensor(onehot)
    
    pred_list = model.generate_z(onehot)
    pred_list = np.array(pred_list)
    
    z_mean = np.mean(pred_list[:,dim], axis=0)
    z_std = np.std(pred_list[:,dim], axis=0)
    print( "Dimension {}: Mean = {}, Standard Deviation = {}".format(dim, z_mean, z_std) )
    
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    
    sns.histplot(pred_list[:,dim], alpha = 0.75 ) # , kde=True, binrange=(-300,300)
    
    ax.set_xlabel('Value', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.title("")
    plt.show()
    plt.savefig( os.path.join(info_dir, "zspace_dim{}.png".format(dim)) )
    

'''
Function5: evaluate the mis-classification for each label in supervised data
input: supervised data file
output: stacked barplot distribution of each class label
'''

def stack_evaluation(model_path, supervise_file, info_dir=None, round_tag="round1"):
    
    model = AAE()
    model.load_state_dict(torch.load( model_path ))
    model.eval()
    
    supervise_data = load_supervise_data(supervise_file, 3, 64, round_tag)
    seqs_list = supervise_data.seqs_list
    label_list = supervise_data.label_list
    nbin = supervise_data.nbin
    
    onehot = seq2onehot(seqs_list, 165)
    onehot = torch.tensor(onehot)
    
    pred_list = model.generate_exp(onehot)
    pred_list = np.array(pred_list)
    
    new_label_list, new_pred_list = [], []
    for item in label_list:
        tmp = [item // nbin, item % nbin]
        new_label_list.append(str(tmp))
    for item in pred_list:
        tmp = [item // nbin, item % nbin]
        new_pred_list.append(str(tmp))
    
    df = pd.DataFrame({"label": new_label_list, "pred": new_pred_list})
    counts = df.groupby('label')['pred'].value_counts().unstack(fill_value=0)
    counts['Sum'] = counts.sum(axis=1)
    counts = counts.sort_values(by='Sum', ascending=False)
    counts = counts.drop("Sum", axis=1)
    
    sns.set(style='darkgrid')
    font = {'size' : 10}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)
    
    counts.plot(kind="bar", stacked=True, alpha=0.7)
    print(counts)
    
    ax.set_xlabel('Value', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.title("")
    plt.show()
    plt.savefig( os.path.join(info_dir, "stackplot_aae_pure.png") )

'''
Function6: evaluate the extrapolation capacity of exogenous sequence
input: exogenous sequence data
output: zspace representation for specified dimension
'''

def extraploation_evaluation(model_path, supervise_file, info_dir=None):
    df_2024 = pd.read_csv("../dataset/AAE_supervised/calibration_mpra_20240201.csv")
    df_2024 = df_2024[~df_2024['tags'].isin(['EConly', 'PAonly', 'ECPA', 'bothNo'])]

    seqs = list(df_2024.loc[:,"seqs"])
    expr_ec = list(df_2024.loc[:,"EC(2024)"])
    expr_pa = list(df_2024.loc[:,"PA(2024)"])

    model = AAE()
    model.load_state_dict(torch.load( model_path ))
    model.eval()

    onehot = seq2onehot(seqs, 165)
    onehot = torch.tensor(onehot)

    nbin = 3
    pred_list = [str([item // nbin, item % nbin]) for item in model.generate_exp(onehot)]

    pred_list = pred_list + pred_list
    expr_list = expr_ec + expr_pa
    label_list = ["EC"] * len(expr_ec) + ["PA"] * len(expr_pa)

    df_boxplot = pd.DataFrame({"prediction": pred_list, "expression": expr_list, "label": label_list})

    sns.set(style='darkgrid')
    font = {'size' : 8}
    matplotlib.rc('font', **font)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 600)

    ax = sns.boxplot( x="prediction", y="expression",  data=df_boxplot,  boxprops=dict(alpha=.9), hue="label", 
                          fliersize=1, flierprops={"marker": 'x'}, palette=["cornflowerblue", "indianred"]) 

    ax.set_xlabel('AAE Prediction', fontsize=10)
    ax.set_ylabel('MPRA Results', fontsize=10)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title("")
    plt.show()
    plt.savefig( os.path.join(info_dir, "boxplot_lib1.png") )


'''
Function7: generate categorical samples and predict their distribution
input: supervised data file
output: prediction barplot for categorical samples
'''

def categorical_evaluation(model_path, supervise_file, info_dir=None, round_tag="round1"):
    
    ### 1. generation model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AAE().to(device)
    model.load_state_dict(torch.load( model_path ))
    model.eval()
    
    for tag in range(0, 8 + 1):
        generated_rep, generated_seq = model.generate_seq(label=tag, batchN=40) 
        generated_rep, generated_seq = np.array(generated_rep), np.array(generated_seq)
        tag_ec = tag // 3
        tag_pa = tag % 3
        write_seq(os.path.join(info_dir, "samples_ec_{}_pa_{}.txt".format(tag_ec, tag_pa)), generated_seq)
    
    ### 2. prediction model (round_tag will modify the model path)
    seqs_list = []
    label_list = []
    pred_ec_list, pred_pa_list = [], []
    
    if round_tag == "round1":
        prefix = "r1"
    elif round_tag == "round2":
        prefix = "final"
    
    model_ec = PredictorNet_language(model_name = "pred_ec", length=165)
    model_ec_path = "./check/pred_ec_{}/checkpoint.pth".format(prefix)
    model_pa = PredictorNet_language(model_name = "pred_pa", length=165)
    model_pa_path = "./check/pred_pa_{}/checkpoint.pth".format(prefix)
    for tag in range(0, 8 + 1):
        tag_ec, tag_pa = tag // 3, tag % 3
        seqs = open_fa( os.path.join(info_dir, "samples_ec_{}_pa_{}.txt".format(tag_ec, tag_pa)) )
        pred_ec = model_ec.predict_input(model_path=model_ec_path, inputs=seqs, mode="data")
        pred_pa = model_pa.predict_input(model_path=model_pa_path, inputs=seqs, mode="data")
        
        seqs_list += seqs
        pred_ec_list += pred_ec
        pred_pa_list += pred_pa
        label_list += [ "EC{}PA{}".format(tag_ec, tag_pa) ] * len(seqs)
    
    df_prediction = pd.DataFrame({"seqs": seqs_list, "pred_ec": pred_ec_list, "pred_pa": pred_pa_list, "labels": label_list})
    df_prediction.to_csv( os.path.join(info_dir, "df_prediction.csv") )
    
    ### boxplot
    df_tmp = pd.read_csv( os.path.join(info_dir, "df_prediction.csv") )
    label = list(df_tmp.loc[:,"labels"])
    pred_ec = list(df_tmp.loc[:,"pred_ec"])
    pred_pa = list(df_tmp.loc[:,"pred_pa"])
    
    label_list = label + label
    pred_list = pred_ec + pred_pa
    spe_list = ["EC"] * len(pred_ec) + ["PA"] * len(pred_pa)
    df_aae_pure = pd.DataFrame({"prediction": pred_list, "label": label_list, "species": spe_list})
    plot_boxplot(os.path.join(info_dir, "boxplot_aae_pure.png") , df_aae_pure )


'''
Function8: generate embedding for 13 species
input: species representative data
output: PCA embedding(representation) of species
'''

def species_evaluation(model_path, supervise_file, info_dir=None):
    
    name_list_13=['Actinobacteria_test0.5','Alphaproteobacteria_test0.5','Bacilli_test0.5', 'Bacteroidia_test0.5',\
            'Betaproteobacteria_test0.5','Clostridia_test0.5','Cyanobacteria_test0.5','Cytophagia_test0.5',\
            'Deltaproteobacteria_test0.5','Flavobacteriia_test0.5','Gammaproteobacteria_test0.5','Negativicutes_test0.5', 'Other_test0.5']

    model = AAE()
    model.load_state_dict(torch.load( model_path ))
    model.eval()

    enc_list_13 = []
    tag_list_13 = []
    data_dir = "../dataset/AAE_represent/"

    for i in range(len(name_list_13)):
        prefix = name_list_13[i]
        data_file = open(data_dir + prefix + ".pickle", "rb+")

        data = pickle.load(data_file)
        data = data

        onehot = seq2onehot(data, 165)
        onehot = torch.tensor(onehot)

        enc = model.generate_rep_unsuper(onehot)
        enc_list_13 += enc
        tag_list_13 += [ prefix.split("_")[0] ] * len(enc)

    X_tsne_13,X_pca_13,X_umap_13 = cal_tSNE_PCA(rep_data=enc_list_13,learning_rate=100,perplexity=12) # 


    name_list_13 = [item.split("_")[0] for item in name_list_13]
    color_list = ["#FF0000", "#FFC0CB", "#D2B48C", "#FFA500", "#FFFF00", "#A52A2A", "#808000", 
                  "#90EE90", "#00FFFF", "#87CEFA", "#0000FF", "#800080", "#D3D3D3"]
    
    plot_species( os.path.join(info_dir, "species13_emb_pca.png") , X_pca_13, tag_list_13, name_list_13, color_list)





        
        
   