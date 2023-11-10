import os
import numpy as np
import math
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import pickle
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering,DBSCAN,OPTICS, cluster_optics_dbscan
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import tensorflow as tf
from ..ProcessData import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

class latent_rep: 
    def __init__(self, 
                 rep_data_7=None,
                 speflag_7=None,
                 rep_dir='./',
                 speflag_3=None,
                 exp_value_3=None,
                 ):
        self.rep_data_7=rep_data_7
        self.speflag_3=speflag_3
        self.speflag_7=speflag_7
        self.exp_value=exp_value_3
        self.rep_dir=rep_dir

    def cal_tSNE_PCA(self,rep_data=None,filename=None,learning_rate=100,perplexity=12):
        print(' [*] Calculating t-SNE and PCA...')
        pca=PCA(n_components=3)
        umaps = umap.UMAP(random_state=21)
        X_tsne = TSNE(n_components=3,learning_rate=100,perplexity=12).fit_transform(rep_data) 
        X_pca = pca.fit_transform(rep_data)
        X_umap = umaps.fit_transform(rep_data)
        with open(self.rep_dir+filename+'_afterPCA.23.2.24.pickle', 'wb') as f:
                    pickle.dump(X_pca, f) 
        return X_tsne,X_pca,X_umap

    def plot_kenrel_density_BS_EC_PA(self, X_pca,name,spe7,epochNstr,AIornaturename="",compareflag=False,spe7_AI=None):
        print(" [*] GAopted data loaded")
        print(' [*] Plotting kernel density of ' +str(name)+'...')
        plot_rep=pd.DataFrame([X_pca[:,0],X_pca[:,1],X_pca[:,2],self.speflag_7])
        plot_rep=pd.DataFrame(plot_rep.values.T, index=plot_rep.columns, columns=['PC_1','PC_2','PC_3','spe'])
        print('plot_rep:')
        print(plot_rep)
        #2d
        plt.figure(5,figsize=(16, 16))
        #3d
        pdf = PdfPages(self.rep_dir+'Dim2_kernel_plot_'+str(name)+'_epoch_'+str(epochNstr)+AIornaturename+'_2.24.pdf')#8.25
        specolor=['Reds','Greens','Blues','Purples','Wistia','Oranges','Greys','winter']#Yellows
        specolor_scatter = ['red','green','blue','purple']
        specolor_AI=['red','green','blue','red','green']
        print(spe7) 
        print('total: '+str(len(self.speflag_7)))
        for i in range(len(spe7)):
            eachspe=spe7[i]
            print(str(eachspe)+':'+str(specolor[i]))
            I=list(np.array(np.where(self.speflag_7==eachspe)).squeeze())
            print(len(I))
            print(plot_rep)

            sns.kdeplot(plot_rep.PC_1[I].astype(float), plot_rep.PC_2[I].astype(float),cmap=specolor[i], shade=False, shade_lowest=False)#shade=False
        if compareflag: 
            for i in range(len(spe7_AI)):
                eachspe=spe7_AI[i]
                print(str(eachspe)+':'+str(specolor_AI[i]))
                I=list(np.array(np.where(self.speflag_7==eachspe)).squeeze())
                print(len(I))
                print(plot_rep)
                if 'naturehigh' in eachspe: 
                    plt.scatter(plot_rep['PC_1'][I],plot_rep['PC_2'][I],c=specolor_AI[i],s=155,marker='^')
                else:
                    plt.scatter(plot_rep['PC_1'][I],plot_rep['PC_2'][I],c=specolor_AI[i],s=185)#marker = 'o' s=65,35
        pdf.savefig() 
        pdf.close()
        return 0

    def plot_dot_BS_EC_PA(self, X_pca,name,spe3,epochNstr):        
        print(' [*] Plotting dot plot of ' +str(name)+'...')
        plot_pca=pd.DataFrame([X_pca[:,0],X_pca[:,1],self.speflag_3,self.exp_value*3+5])#+5,+7
        plot_pca=pd.DataFrame(plot_pca.values.T, index=plot_pca.columns, columns=['PC_1','PC_2','FunctionalSpe','exp'])
        plt.figure(figsize=(16, 16))
        plot_pca.to_csv(self.rep_dir+'dotplot_'+name+'.csv')
        pdf = PdfPages(self.rep_dir+'Dim2_dot_plot_'+str(name)+'_epoch_'+str(epochNstr)+'.pdf')
        cm_list=[plt.cm.get_cmap('Reds'),plt.cm.get_cmap('Blues'),plt.cm.get_cmap('Greens'),'lightgrey']# #Wistia
        spe3_3only=spe3[:3]
        spe3_3only.append(spe3[6])
        spe3=spe3_3only
        for i in range(len(spe3)):
            i = 3-i
            eachspe=spe3[i]
            colors=cm_list[i]
            print(str(eachspe)+':'+str(colors))
            I=list(np.array(np.where(self.speflag_3==eachspe)).squeeze())
            print(len(I))
            if 'only' in eachspe:
                plt.scatter(plot_pca['PC_1'][I],plot_pca['PC_2'][I],c=plot_pca['exp'][I], vmin=min(plot_pca['exp'][I])-1, vmax=max(plot_pca['exp'][I])+1, s=35, cmap=colors)# vmin=0, vmax=20
            else:
                plt.scatter(plot_pca['PC_1'][I],plot_pca['PC_2'][I],c=colors)
        pdf.savefig() 
        pdf.close()
        return 0
        
    ####2.
    def get_quantitative_clustering_result(self,n_clusters = 8,connectmatrix=True,method_list=['AgglomerativeClustering','DBSCAN','OPTICS'],spe_only3=['onlyEC_3_test20','onlyBS_3_test20','onlyPA_3_test20']):
        st = time.time()
        n_neighbor=30 
        print(" [*] Compute structured hierarchical clustering...")
        X=self.rep_data_7
        #
        cal_only3_64data=[]
        cal_only3_64spe=[]
        for i in range(len(spe_only3)):
            eachspe=spe_only3[i]
            I=list(np.array(np.where(self.speflag_7==eachspe)).squeeze())
            print(len(I))
            cal_only3_64data.append(self.rep_data_7[I])
            cal_only3_64spe.append(self.speflag_7[I])
        cal_only3_64data=np.concatenate(cal_only3_64data)
        cal_only3_64spe=np.concatenate(cal_only3_64spe)
        print(cal_only3_64data.shape)
        print(cal_only3_64spe.shape)
        #ARI,AMI,FMS:[0,1]
        t0 = time.time()
        for method in method_list:
            if method=='AgglomerativeClustering':
                print('Cal '+str(method)+' ...')
                labels_pred=self.calAgglomerativeClustering(cal_only3_64data,n_clusters,n_neighbor,connectmatrix)
                ARI,AMI,FMS=self.cal_criteria(cal_only3_64spe,labels_pred,'AgglomerativeClustering')
            elif method=='DBSCAN':
                print('Cal '+str(method)+' ...')
                labels_pred=self.cal_DBSCAN(X=cal_only3_64data,eps=0.1,min_samples=32)#X=self.rep_data_7 32 ########1.eps
                ARI,AMI,FMS=self.cal_criteria(cal_only3_64spe,labels_pred,'DBSCAN')
            elif method=='OPTICS':
                print('Cal '+str(method)+' ...')
                labels_pred=self.cal_OPTICS(X=cal_only3_64data,xi=0.5,min_samples=32,min_cluster_size=0.5)#X=self.rep_data_7
                ARI,AMI,FMS=self.cal_criteria(cal_only3_64spe,labels_pred,'OPTICS')
        #
        criteria=['Adjusted Rand Index','Adjusted Mutual Information','Fowlkes Mallows scores']
        value=[ARI,AMI,FMS]
        plot_cluster=pd.DataFrame([value,criteria]) 
        self.plot_cluster=pd.DataFrame(plot_cluster.values.T,index=plot_cluster.columns,columns=['value','criteria'])
        clusterfile='plot_cluster_7_'+method+'.pickle'
        with open(self.rep_dir+clusterfile, 'wb') as f:
            pickle.dump(plot_cluster, f)
        print('saved!')
        return 
    def cal_criteria(self,label_real,label_pred,method):
        ARI=metrics.adjusted_rand_score(label_real, label_pred)
        AMI=metrics.adjusted_mutual_info_score(label_real, label_pred)
        FMS=metrics.fowlkes_mallows_score(label_real, label_pred)
        V_measure_score=metrics.v_measure_score(label_real, label_pred)
        print(method+'_ARI: '+str(ARI))
        print(method+'_AMI: '+str(AMI))
        print(method+'_FMS: '+str(FMS))
        print(method+'_V_measure_score'+str(V_measure_score))
        return ARI,AMI,FMS
        
    def cluster_plot(self):
        print(" [*] Plot clustering result...")
        plt.figure(figsize=(16, 12))
        pdf = PdfPages(self.rep_dir+'rep_layers_cluster_evlau.pdf')
        print(self.plot_cluster)
        sns.barplot(x="criteria",y="value", data=self.plot_cluster)
        pdf.savefig()
        pdf.close()
        return 

    def calAgglomerativeClustering(self,X,n_clusters=7,n_neighbor=30,connectmatrix=True):
        if connectmatrix:
            knn_graph = kneighbors_graph(X, n_neighbor, include_self=False) 
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=knn_graph)
        else:
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        ward.fit(X)
        labels_pred = np.array(ward.labels_)
        return labels_pred

    def cal_DBSCAN(self,X,eps,min_samples):
        db=DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels_pred=np.array(db.labels_)
        return labels_pred

    def cal_OPTICS(self,X,xi=.05,min_samples=50,min_cluster_size=.05):
        optics=OPTICS(min_samples=min_samples, cluster_method='xi', xi=xi, min_cluster_size=min_cluster_size).fit(X)
        labels_pred=np.array(optics.labels_)
        return labels_pred
        
    def plot_PCA_3D(self,X_pca,name,spe3,PCdim_list=[[1,2,3]]):
        print(' [*] Plotting 3D plot of rep' +str(name)+'...')
        print(X_pca.shape)
        for i in range(len(PCdim_list)):
            PCdim=PCdim_list[i]
            PCdimstr=''.join(list(map(str, PCdim)))
            print('Ploting PCA '+PCdimstr+'...')
            fig2 = plt.figure(figsize=(20, 10))
            pdf = PdfPages(self.rep_dir+str(name)+'_3D_'+PCdimstr+'.pdf')
            PC1=np.array(X_pca[:,PCdim[0]]) #(2388,)
            PC2=np.array(X_pca[:,PCdim[1]])
            PC3=np.array(X_pca[:,PCdim[2]])
            ax = Axes3D(fig2)
            print('spe3:')
            print(spe3)
            if '3' in 'name':
                speflag=self.speflag_3
            else:
                speflag=self.speflag_7
            cm_list=plt.cm.Spectral_r(np.linspace(0, 1, len(spe3)))
            for j in range(len(spe3)): 
                I=list(np.array(np.where(speflag==spe3[j])).squeeze())
                ax.scatter(PC1[I], PC2[I], PC3[I], c=cm_list[j], marker='.',s=60,label=spe3[j])
            ax.legend(loc='upper left', fontsize=15,framealpha=1,ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
            ax.set_xlabel('PC'+str(PCdim[0]))
            ax.set_ylabel('PC'+str(PCdim[1]))
            ax.set_zlabel('PC'+str(PCdim[2]))
            pdf.savefig() 
            pdf.close()
        return