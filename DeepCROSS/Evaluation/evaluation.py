import os
import sys
import shutil
import time
import math
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import scipy.stats
from scipy.stats import pearsonr
import pickle
from matplotlib.backends.backend_pdf import PdfPages
pd.set_option('display.max_columns',None)
# from ..Dataloader import *
# from ..Generators import *


#GC,K-mer,ployN,similarity
class evalu: 
    def __init__(self, 
                 data_dir='./HierDNA/hier_input/',
                 save_dir='./HierDNA/hier_output/'
                 ):
        self.data_dir=data_dir
        self.save_dir=save_dir

    def JS_divergence(self,p,q):
        M=(p+q)/2
        return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

    def remove_nan_inf(self,data,col):
        for i in range(len(data)):
            if str(data[col][i])=='nan' or str(data[col][i])=='inf':
                data[col][i]=0
        return data

    def seq_GC(self,seq):
        GCcount=0
        for i in range(len(seq)): 
            if(seq[i])=='G'or (seq[i])=='C':
                GCcount+=1
        return GCcount/len(seq)*100

    def GC_content(self,filename):
        fasta_file = os.path.join(self.data_dir,filename) 
        with open(fasta_file, 'r') as f:
            GC=[]
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    GC.append(self.seq_GC(seq))
        return GC 

    def GC_plot_9boxplot(self,evalu_file,compare_file_list,sample_dir):
        print(' [*] Plot GC criterion...')
        color_list=['orange','blue','black','brown','darkblue','purple','grey','yellow','green','red','orange','green']
        plt.figure(figsize=(60, 20))
        pdf = PdfPages(self.save_dir+'/GC_content_3boxplot_'+evalu_file.strip('.txt')+'.pdf')
        sns.set(font_scale = 2)
        GC_result_gen = self.GC_content(evalu_file)
        for i in range(len(compare_file_list)):
            genlen=len(GC_result_gen)
            compare_file = compare_file_list[i]
            source1 = './'+compare_file 
            evalu_file=evalu_file.split('.')[0] 
            target = sample_dir 
            shutil.copy(source1, target)   
            GC_result_nature=self.GC_content(compare_file)
            naturelen = len(GC_result_nature)
            GC_result_nature.extend(GC_result_gen)
            gen_nature=GC_result_nature 
            huelist = ['nature']*naturelen
            huelist.extend(['gen']*genlen)
            GC_result = pd.DataFrame([gen_nature,huelist]).T
            compare_content=compare_file.split('_paper')[0]
            GC_result.columns = ['gen_nature',compare_content]
            ax = plt.subplot(1,3,i+1, frameon = False) 
            sns.boxplot(x=compare_content,y="gen_nature",data=GC_result,palette="colorblind",width=0.5,linewidth=5)#Set2
            plt.ylim(0, 80)
        plt.legend()
        pdf.savefig() 
        pdf.close()


    def GC_plot_violin_groupmethod(self,evalu_file_list,compare_file_list,outpdf_name):
        print(' [*] Plot GC criterion...')
        method_list = ['EConly','PAonly','EC&PA']
        GC_nature_gen = []
        huelist=[]
        huelist2=[]
        exp_situation_group=[]
        exp_situation_group2=[]
        js_dist=[]
        for i in range(3):
            evalu_file_situation=evalu_file_list[i]
            compare_file = compare_file_list[i]
            GC_result_nature = self.GC_content(compare_file)
            naturelen = len(GC_result_nature)
            GC_nature_gen.extend(GC_result_nature)
            compare_content=compare_file.split('_test20')[0]
            huelist.extend([compare_content]*naturelen)
            exp_situation_group.extend([method_list[i]]*naturelen)
            for j in range(4):
                evalu_file=evalu_file_situation[j]
                GC_result_gen = self.GC_content(evalu_file)
                genlen=len(GC_result_gen)
                GC_nature_gen.extend(GC_result_gen)
                if j<3:
                    huelist.extend([evalu_file.split('_method_')[1].strip('.txt')]*genlen)
                    huelist2.extend([evalu_file.split('_method_')[1].strip('.txt')])
                else:
                    huelist.extend([evalu_file.strip('.txt')]*genlen)
                    huelist2.extend([evalu_file.strip('.txt')])
                exp_situation_group.extend([method_list[i]]*genlen)
                exp_situation_group2.extend([method_list[i]])
                eachJS = self.cal_JS(GC_result_gen,GC_result_nature)
                js_dist.append(eachJS)
        #violin plot
        GC_result = pd.DataFrame([GC_nature_gen,huelist,exp_situation_group]).T
        GC_result.columns = ['gen_nature_GCvalue','method','Exp_situation_group']
        GC_result['gen_nature_GCvalue'] = GC_result['gen_nature_GCvalue'].astype(float)
        plt.figure(figsize=(60, 20))
        sns.set(font_scale = 2)
        pdf = PdfPages(self.save_dir+outpdf_name+'voilinplot.pdf')
        sns.set_style("white")
        sns.violinplot(x="Exp_situation_group", y="gen_nature_GCvalue", hue="method",data=GC_result, palette="muted")
        sns.despine()
        pdf.savefig() 
        pdf.close()

        #bar plot 
        js_dist = np.array(js_dist)
        GC_result2 = pd.DataFrame([js_dist,huelist2,exp_situation_group2]).T
        GC_result2.columns = ['JSdist','method','Exp_situation_group']
        GC_result2['JSdist'] = GC_result2['JSdist'].astype(float)
        #
        print('GC_result2:')
        print(GC_result2)
        plt.figure(figsize=(20, 20))
        sns.set(font_scale = 2)
        pdf = PdfPages(self.save_dir+outpdf_name+'barplot.pdf')
        sns.set_style("white")
        sns.barplot(x="Exp_situation_group", y="JSdist",hue="method",data=GC_result2,palette="muted")        #sns.set_style("white")
        sns.despine()
        pdf.savefig() 
        pdf.close()
        return 

    def GC_plot_9andbinJSdistance(self,evalu_file_list,compare_file_list,outpdf_name):
        method_list = ['onlysupervise','supervise_local','final_model']
        for i in range(3):
            evalu_file_method=evalu_file_list[i]
            js_dist=[]
            for j in range(3):
                compare_file = compare_file_list[j]
                GC_result_nature=self.GC_content(compare_file)
                for k in range(len(evalu_file_method)):
                    eachJS=[]
                    evalu_file = evalu_file_method[k]
                    GC_result_gen = self.GC_content(evalu_file)
                    GC_result_nature = np.array(GC_result_nature)
                    GC_result_gen = np.array(GC_result_gen)
                    n = min(len(GC_result_gen),len(GC_result_nature)) 
                    if len(GC_result_nature)> n:
                        seq_index_A = np.arange(GC_result_nature.shape[0])
                        for k in range(3):
                            np.random.seed(k+2)
                            np.random.shuffle(seq_index_A)
                            GC_result_nature2 = GC_result_nature[seq_index_A[:n]]
                            eachJS.append(self.JS_divergence(GC_result_nature2,GC_result_gen))
                    else:
                        seq_index_A = np.arange(GC_result_gen.shape[0])
                        for k in range(3):
                            np.random.seed(k+2)
                            np.random.shuffle(seq_index_A)
                            GC_result_gen2 = GC_result_gen[seq_index_A[:n]]
                            eachJS.append(self.JS_divergence(GC_result_nature,GC_result_gen2))
                    eachJS=np.mean(np.array(eachJS))
                    js_dist.append(eachJS)
            js_dist_result = np.array(js_dist).reshape((3,3)) 
            js_dist_result = pd.DataFrame(js_dist_result,index=['AI EConly','AI PAonly','AI EC&PA'],columns=['EConly','PAonly','EC&PA'])
            plt.figure(figsize=(60, 48))
            pdf = PdfPages(self.save_dir+outpdf_name+method_list[i]+'.pdf')
            sns.heatmap(data=js_dist_result,cmap=plt.get_cmap('Greens'),vmin=0.006, vmax=0.015)
            plt.legend()
            pdf.savefig() 
            pdf.close()
        return 

    def cal_JS(self,GC_result_gen,GC_result_nature):
        GC_result_nature = np.array(GC_result_nature)
        GC_result_gen = np.array(GC_result_gen)
        n = min(len(GC_result_gen),len(GC_result_nature))
        eachJS=[]
        if len(GC_result_nature)> n:
            seq_index_A = np.arange(GC_result_nature.shape[0])
            for k in range(3):
                np.random.seed(k+2)
                np.random.shuffle(seq_index_A)
                GC_result_nature2 = GC_result_nature[seq_index_A[:n]]
                eachJS.append(self.JS_divergence(GC_result_nature2,GC_result_gen))
        else:
            seq_index_A = np.arange(GC_result_gen.shape[0])
            for k in range(3):
                np.random.seed(k+2)
                np.random.shuffle(seq_index_A)
                GC_result_gen2 = GC_result_gen[seq_index_A[:n]]
                eachJS.append(self.JS_divergence(GC_result_nature,GC_result_gen2))
        eachJS=np.mean(np.array(eachJS))
        return eachJS





    #2.
    def polyN_seq(self,seq,min,max):
        polyN=np.zeros(shape=((max-min+1),1))
        kmern=np.zeros(shape=((max-min+1),1))
        baselist=['A','G','C','T']
        for n in range(min,max+1,1):
            kmern[n-min,]+=len(seq)-n+1
            for i in range(len(seq)-n+1):
                for base in baselist:
                    if seq[i:i+n]==str(base)*n:
                        polyN[n-min,]+=1
        return kmern,polyN

    def polyN_count(self,filename,min,max):
        fasta_file = self.data_dir+'/'+filename
        with open(fasta_file, 'r') as f:
            kmern_allseq=np.zeros(shape=((max-min+1),1))
            polyN_count=np.zeros(shape=((max-min+1),1))
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    kmern,polyN_count_add=self.polyN_seq(seq,min,max)
                    polyN_count+=polyN_count_add
                    kmern_allseq+=kmern
        return np.array(polyN_count)/kmern_allseq

    def polyN_plot(self,species_list,color_list,min,max):
        print(' [*] Plot ployN criterion...')
        pdf = PdfPages(self.save_dir+'/PolyN_'+str(species_list[1])+'.pdf') 
        plt.figure(figsize=(7,7)) 
        data=pd.DataFrame([])
        for i in range(len(species_list)):
            spe=species_list[i]
            filename= spe+'.txt'
            PloyN_result=self.polyN_count(filename,min,max).reshape((-1))
            PloyN_result=pd.DataFrame([PloyN_result,np.array([str(spe)]*(max-min+1))]).T
            data=pd.concat([data,PloyN_result])
        data.columns=['number','Species']
        sns.barplot(x=np.array(data.index)+min,y='number',hue='Species',data=data,
                    alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel('PolyN')
        plt.ylabel('Frequency')
        pdf.savefig() 
        pdf.close()
    #3.
    def count_kmer(self,seq,kmers,numk):
        k = numk
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N'not in kmer:
                if kmer in kmers:
                    kmers[kmer] += 1
                else:
                    kmers[kmer] = 1
        sortedKmer = sorted(kmers.items(), reverse=True)
        return sortedKmer,kmers
            
    def kmer_fasta(self,filename,numk):
        input_fasta_file=os.path.join(self.data_dir,filename)#path+'data/'+filename
        kmer_result= pd.DataFrame([])
        count=[]
        kmerseq=[]
        with open(input_fasta_file, 'r') as f:
            seq = ""
            kmers={}
            for line in f.readlines():
                if not line.startswith(">"):
                    seq = line.strip()
                    sortedKmer,kmer=self.count_kmer(seq,kmers,numk)
        for key,value in sortedKmer:
            kmerseq.append(key)
            count.append(value)
        kmer_result['kmer']=kmerseq
        kmer_result['count']=count
        kmer_sort_plot=kmer_result.sort_values(by=['count'])
        return kmer_sort_plot

    def kmer_statistic(self,species_list,method,numk=6):
        data=pd.DataFrame([])
        for i in range(len(species_list)):
            spe=species_list[i]
            filename= spe+'.txt'
            kmer_sort_plot=self.kmer_fasta(filename,numk)
            kmer_sort_plot.columns=['kmer',str(spe)]
            if data.shape!=(0, 0):
                data=pd.merge(data,kmer_sort_plot,how='left',on='kmer')
            else:
                data=kmer_sort_plot
            kmer_compare=self.remove_nan_inf(data,str(spe))
            print('kmer_compare:')
            print(kmer_compare) 
        f = open(os.path.join(self.save_dir,'kmer_result_'+str(numk)+'mer'+'_'+method+'.pkl'), 'wb') #open(self.save_dir+'/kmer_result_'+str(species_list[1])+'_'+str(numk)+'.pkl', 'wb')
        pickle.dump(kmer_compare, f)
        print(' [*] Evalu kmer cal done')
        f.close()

    def kmer_plot_heatmap(self,draw_list_all,method,numk=6):
        print(' [*] kmer correlation heatmap...')
        plt.figure(figsize=(60, 60))
        pdf = PdfPages(self.save_dir+'/kmer_heatmap_9'+'_'+str(numk)+'mer_'+method+'.pdf') 
        r_9 =[]
        for i in range(len(draw_list_all)):
            draw_list = draw_list_all[i]
            sort_by_species = draw_list[1]
            if draw_list[0]==draw_list[1]:
                sort_by_species=sort_by_species+'_x'
                draw_list[0]=draw_list[0]+'_x'
                draw_list[1]=draw_list[1]+'_y'
            kmer_compare = pickle.load(open(os.path.join(self.save_dir,'kmer_result_'+str(numk)+'mer'+'_'+method+'.pkl'), 'rb'))
            kmer_compare=kmer_compare.sort_values(by=[str(sort_by_species)])#Storz#,ascending=False
            kmer_compare=kmer_compare.reset_index(drop=True)
            y = locals()
            for j in range(len(draw_list)):
                y[j]=np.array(kmer_compare[draw_list[j]])/np.sum(np.array(kmer_compare[draw_list[j]]))*100
            r=pearsonr(y[0],y[1])[0]  
            r=("%.3f" % r )
            r_9.append(r)
        r_9 = np.array(r_9).reshape((3,3))
        r_9 = pd.DataFrame(r_9,index=['AI EConly','AI PAonly','AI EC&PA'],columns=['EConly','PAonly','EC&PA'])
        r_9['EConly'] = r_9['EConly'].astype(float)
        r_9['PAonly'] = r_9['PAonly'].astype(float)
        r_9['EC&PA'] = r_9['EC&PA'].astype(float)
        print('r_9:')
        print(r_9)
        sns.heatmap(data=r_9,cmap=plt.get_cmap('Greens'))
        plt.legend()
        pdf.savefig() 
        pdf.close() 
        return 

    def kmer_plot_corplot(self,draw_list_all,method,numk=6):
        print(' [*] kmer correlation corplot...')
        plt.figure(figsize=(60, 60))
        pdf = PdfPages(self.save_dir+'/kmer_cor_9'+'_'+str(numk)+'mer_'+method+'.pdf') 
        for i in range(len(draw_list_all)):
            draw_list = draw_list_all[i]
            sort_by_species = draw_list[1]
            if draw_list[0]==draw_list[1]:
                sort_by_species=sort_by_species+'_x'
                draw_list[0]=draw_list[0]+'_x'
                draw_list[1]=draw_list[1]+'_y'
            kmer_compare = pickle.load(open(os.path.join(self.save_dir,'kmer_result_'+str(numk)+'mer'+'_'+method+'.pkl'), 'rb'))
            kmer_compare=kmer_compare.sort_values(by=[str(sort_by_species)])#Storz#,ascending=False
            kmer_compare=kmer_compare.reset_index(drop=True)
            y = locals()
            for j in range(len(draw_list)):
                y[j]=np.array(kmer_compare[draw_list[j]])/np.sum(np.array(kmer_compare[draw_list[j]]))*100
            r=pearsonr(y[0],y[1])[0]  
            r=("%.3f" % r )
            ax = plt.subplot(3,3,i+1, frameon = False) 
            plt.scatter(y[0],y[1],color=str('blue'),s=7,alpha=0.5,marker='.')#color_list[0]
            plt.xlim(min(min(y[0]),min(y[1]))-0.02,max(max(y[0]),max(y[1])))
            plt.ylim(min(min(y[0]),min(y[1]))-0.02,max(max(y[0]),max(y[1])))
            plt.text(max(y[0])*0.8,max(y[1])*0.8,r'r='+str(r))
            plt.xlabel(str(draw_list[0]))
            plt.ylabel(str(draw_list[1]))
        pdf.savefig() 
        pdf.close()  
        return 

    def kmer_plot_barplot(self,draw_list_all,numk=6):
        print(' [*] kmer correlation barplot...')
        plt.figure(figsize=(60, 60))
        pdf = PdfPages(self.save_dir+'/kmer_barplot_9_'+str(numk)+'mer.pdf') 
        r_9 =[]
        for i in range(len(draw_list_all)):
            draw_list = draw_list_all[i]
            print(draw_list)
            sort_by_species = draw_list[1]
            kmer_compare = pickle.load(open(os.path.join(self.save_dir,'kmer_result_'+str(numk)+'mer'+'_all.pkl'), 'rb'))
            kmer_compare=kmer_compare.sort_values(by=[str(sort_by_species)])#Storz#,ascending=False
            kmer_compare=kmer_compare.reset_index(drop=True)
            print(kmer_compare.columns)
            y = locals()
            for j in range(len(draw_list)):
                y[j]=np.array(kmer_compare[draw_list[j]])/np.sum(np.array(kmer_compare[draw_list[j]]))*100
            r=pearsonr(y[0],y[1])[0]  
            r=("%.3f" % r )
            r_9.append(r)
        method = ['method1','method2','method3','nature']*3
        Exp_situation_group = ['EC','EC','EC','EC','PA','PA','PA','PA','both','both','both','both']
        r9_result = pd.DataFrame([r_9,method,Exp_situation_group]).T
        r9_result.columns = ['r_9','method','Exp_situation_group']
        r9_result['r_9'] = r9_result['r_9'].astype(float)
        #
        print('r9_result:')
        print(r9_result)
        plt.figure(figsize=(20, 20))
        sns.set(font_scale = 2)
        sns.set_style("white")
        sns.barplot(x="Exp_situation_group", y="r_9",hue="method",data=r9_result,palette="muted")
        plt.ylim(0,1)
        sns.despine()
        pdf.savefig() 
        pdf.close()
        return 

    #4.
    def kmer_loc(self,species_list,pattern,seqlen):
        filenum=len(species_list)
        k=len(pattern)
        position_count=np.zeros([seqlen-k+1,filenum],dtype='int')
        for i in range(len(species_list)):
            fasta_file=species_list[i]
            f=open(self.save_dir+'data/'+fasta_file+'_promoter_fa.txt')
            for seq in f.readlines():
                if '>' not in seq:
                    seq=seq[:-1]
                    for j in range(len(seq) - k + 1):
                        if seq[j:j+k]==pattern:
                            position_count[j,i]+=1
            f.close()  
        position_count=pd.DataFrame(position_count)
        position_count.columns=species_list
        return position_count

    def kmer_location_plot(self,species_list,color_list,pattern,seqlen,numk=6):
        kmer_location=self.kmer_loc(self.save_dir,species_list,pattern,seqlen)
        x=np.array(range(kmer_location.shape[0]))
        y=locals()
        plt.figure(figsize=(6,6)) 
        pdf = PdfPages(self.save_dir+'result_pdf/'+pattern+'_location_'+str(numk)+'mer.pdf') 
        for i in range(len(species_list)):
            spe=species_list[i]
            y[i]=np.array(kmer_location[str(spe)])/np.sum(np.array(kmer_location[str(spe)]))*100
            plt.plot(x,y[i],color=color_list[i],alpha=0.5,marker='.',label=str(spe))   
        plt.legend(loc='upper left', fontsize=10,framealpha=0,ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
        plt.xlabel(str(pattern))
        plt.ylabel('Frequency(%)')
        pdf.savefig() 
        pdf.close()
        return

    def find_kmer_time(self,whichdataset,pattern,numk=6):
        kmer_compare = pickle.load(open(self.save_dir+'kmer_result_'+str(numk)+'.pkl', 'rb'))
        kmer_compare=kmer_compare.sort_values(by=[whichdataset])#,ascending=False
        kmer_compare=kmer_compare.reset_index(drop=True)
        location=0
        for i in range(len(kmer_compare)):
            if kmer_compare['kmer'][i]==str(pattern):
                location=i
                break
        return location
    
    #5.
    def latent_similarity_preserve(self,EncoderNet,DecoderNet,step=10,sample_number=10): #
        print(' [*] Plotting latent space similarity preserve...')
        name_list=['univ_random1','onlyBS_random1','onlyEC_random1','onlyPA_random1','BS_EC_random1','BS_PA_random1','EC_PA_random1']
        sp_data=[]
        speflag=[]
        plt.figure(figsize=(10, 20))
        pdf = PdfPages(self.save_dir+'latent_similarity_stepout_hammingdist.pdf')
        for i in range(len(name_list)):
            fname=name_list[i]
            print(fname)
            with open('./data/seq_7_random1_pickle/'+fname+'.pickle','rb') as f:
                promoter = pickle.load(f)
            charmap,invcharmap = GetCharMap(promoter)
            input_oh = torch.from_numpy(seq2oh(promoter,charmap))
            input_oh = np.swapaxes(input_oh,1,2)
            with torch.no_grad(): 
                z=EncoderNet(input_oh.cuda().type(torch.cuda.FloatTensor)).cpu().numpy() #(1,64)
            hamm_similarity_list=[]
            reconstructseq=open(self.save_dir+'/reconstructseq_7.txt','w')
            for i in range(step):
                if i==0:
                    with torch.no_grad(): 
                        surrounding_seq=self.Generate_seq(batch_size=64,z_dim=64,invcharmap=['T', 'C', 'G', 'A'],DecoderNet=DecoderNet)
                    reconstructseq.write('>'+fname[:-8]+'\n')
                    reconstructseq.write(str(surrounding_seq[0])+'\n')
                else:
                    cov = np.eye(64)*(0.1*i)
                    surrounding_z = np.random.multivariate_normal(z.squeeze(),cov,sample_number) 
                    with torch.no_grad(): 
                        surrounding_seq=self.Generate_seq(batch_size=64,z_dim=64,invcharmap=['T', 'C', 'G', 'A'],DecoderNet=DecoderNet)
                #mean of hamming distance
                hamm_dist_mean=self.cal_hamming(promoter,surrounding_seq) 
                hamm_similarity_list.append(165-hamm_dist_mean)
            plt.plot(range(step),hamm_similarity_list,label=str(fname[:-8]))
        plt.legend(loc='upper right')
        pdf.savefig() 
        pdf.close()
        reconstructseq.close()
        print(' [Done] latent_similarity_preserve')
        return

    def cal_hamming(self,promoter,surrounding_seq):
        promoter=self.seq2compare(promoter).squeeze()
        surrounding_seq=self.seq2compare(surrounding_seq)
        promoter=np.tile(promoter,len(surrounding_seq)).reshape(len(surrounding_seq),-1)
        hamm_dist_mean=np.sum(promoter!=surrounding_seq)/len(surrounding_seq)
        return hamm_dist_mean


    def seq2compare(self,seqs): 
        Length=len(seqs[0])
        map={'A':0,'G':1,'C':2,'T':3}
        seqall=[]
        for seq in seqs:
            eachseq=np.zeros([Length],dtype = 'float')
            for j in range(len(seq)):
                eachseq[j]=map[seq[j]]
            seqall.append(eachseq)
        return np.array(seqall)

    def Generate_seq(self,batch_size,z_dim,invcharmap,DecoderNet=None,z=None,outputform='seq'):#隐空间变为序列(oh/str)
        if z is None:
            z = np.random.normal(size=(batch_size*100,z_dim))
        num = z.shape[0]
        batches = math.ceil(num/batch_size)
        generated_seq = []
        ohs = []
        z=torch.from_numpy(z)
        for b in range(batches):
            with torch.no_grad(): 
                oh = DecoderNet(z[b*batch_size:(b+1)*batch_size].cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
            ohs.extend(oh)
            oh = np.swapaxes(oh,1,2)
            eachseq=oh2seq(oh,invcharmap)
            generated_seq.extend(eachseq)
        if outputform == 'oh':
            return np.array(ohs)
        else:
            return generated_seq




















###############
#save code
###############
    def kernels_nomalize(self,kernels):#kernel normalize为count matrix
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        return kernels

    def RC(self,seq):
        seqout=''
        basedic={'A':'T','G':'C','C':'G','T':'A'}
        for i in range(len(seq)):
            seqout+=basedic[seq[len(seq)-1-i]]
        return seqout

    #Rigidity_and_Motif评估
    def Rigidity_and_Motif(self):
        self.model.eval()
        #for name in self.model.state_dict():
            #print(name)
        #PWM
        #第一层：PWM kernels(32,4,7):32个不同的kernel,4碱基onehot,7个碱基长度的motif
        PWM_kernels_1 = self.model.renderer.PWM_scan[0].weight.cpu().detach().clone()  #renderer.conv1.0: conv1
        self.PWM_out_TOMTOM(PWM_kernels_1)
        #第二层：PWM_combination kernel(64*32*3),应该看刺激这个kernel的是什么信号
        PWM_signal_2 = self.model.renderer.PWM_combination[0].weight.cpu().detach().clone() 
        #Rigidity
        #第一层：Rigidity kernels(32,4,7):32个不同的kernel,4碱基onehot,7个碱基长度的特征
        Rigidity_kernels_1 = self.model.renderer.Ridity_conv1[0].weight.cpu().detach().clone()  #renderer.conv1.0: conv1
        self.Rigidity_out_Tetranucleotide(Rigidity_kernels_1,rigidity_file='Tetranucleotide_136_Flexibility.csv')
        self.Rigidity_out_Tetranucleotide(Rigidity_kernels_1,rigidity_file='Tetranucleotide_136_minenergy.csv')
        #第二层：Rigidity kernels
        Rigidity_kernels_2 = self.model.renderer.Ridity_conv2[0].weight.cpu().detach().clone()  #renderer.conv1.0: conv1
        #self.Rigidity_out_Tetranucleotide(Rigidity_kernels_2)
        print('[Done] Rigidity and motif evaluation')
        return

    def Rigidity_out_Tetranucleotide(self,kernels,rigidity_file='Tetranucleotide_136_Flexibility.csv'):
        #out_forTetra=open(self.save_dir+'kernel_vis_Tetranucleotide.txt','w')
        bycol=rigidity_file.split('_')[2][:-4]
        flexi_data=pd.read_csv(self.data_dir+rigidity_file).sort_values(axis=0,ascending=True,by=[bycol])
        index=0
        Rankout=open(self.save_dir+bycol+'_Rank_out.txt','w')
        bases=['A','G','C','T']
        Tetras=[]
        Ranks=[]
        for i in range(kernels.shape[0]):
            Tetra=''
            matrix = np.array(kernels[i]) #4*7
            baseIs=np.argmax(matrix, axis=0)#np.where(matrix==np.max(matrix,axis=0))[0]
            for baseI in baseIs:
                Tetra+=bases[baseI]
            if Tetra in list(flexi_data['Bases']):
                for i in range(len(flexi_data)):
                    if Tetra==str(flexi_data['Bases'][i]):
                        index+=1
                        Rankout.write('>'+'kernel_'+str(index)+'\n')
                        Rankout.write(str(i+1)+'\n')
                        if Tetra not in Tetras:
                            Tetras.append(Tetra)
                            Ranks.append((i+1)/136)
            elif self.RC(Tetra) in list(flexi_data['Bases']):
                for i in range(len(flexi_data)):
                    if self.RC(Tetra)==str(flexi_data['Bases'][i]):
                        index+=1
                        Rankout.write('>'+'kernel_'+str(index)+'_'+Tetra+'\n')
                        Rankout.write(str(i+1)+'\n')
                        if Tetra not in Tetras:
                            Tetras.append(Tetra)
                            Ranks.append((i+1)/136)
            else:
                index+=1
                print(Tetra)
                print('Not in: '+rigidity_file)
        Rankout.close()
        plotdata=pd.DataFrame([Tetras,Ranks]).T
        plotdata.columns=['Tetras','Ranks']
        plotdata=plotdata.sort_values(axis=0,ascending=True,by=['Ranks'])
        print("[*] Reconstruct Loss ploting...")
        print('Max rank is '+str(np.max(Ranks)))
        plt.figure(1,figsize=(12, 6))
        pdf = PdfPages(self.save_dir+'Rigidity_'+bycol+'.pdf')
        sns.barplot(data=plotdata,x='Tetras', y='Ranks', saturation=0.2)
        plt.ylim(0,1)
        pl.xticks(rotation=90)
        pdf.savefig() 
        pdf.close()
        plt.clf() 
        plt.cla()
        plt.close()
        return

    def PWM_out_TOMTOM(self,kernels):
        out_forTOMTOM=open(self.save_dir+'kernel_vis_motif.txt','w')
        bases=['A','G','C','T']
        index=1
        for i in range(kernels.shape[0]):
            matrix = np.array(kernels[i]) #4*7
            out_forTOMTOM.write('>'+'Kernel_'+str(index)+'\n')
            index+=1
            for k in range(len(matrix)):
                line=matrix[k]
                line=list(np.array(line*100,dtype='int32'))
                for j in range(len(line)):
                    if line[j]<0: line[j]=0
                line=[bases[k]]+line
                out_forTOMTOM.write(str(line)[1:-1].replace(",", "  ").replace("'","")+'\n')
        out_forTOMTOM.close()
        return

