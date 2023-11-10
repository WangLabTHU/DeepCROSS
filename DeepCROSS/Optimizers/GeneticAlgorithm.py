import numpy as np
import os
import math
from ..ProcessData import seq2oh,oh2seq,saveseq,GetCharMap
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')


class GeneticAthm():
    def __init__(self,
                 Generator,
                 Predictor,
                 Valid = None,
                 z_dim=64):
        self.z_dim = z_dim
        self.x = np.random.normal(size=(3200,z_dim))
        self.Generator = Generator
        try:
            self.Seqs = self.Generator(self.x)
            if type(self.Seqs) is not list or type(self.Seqs[0]) is not str:
                print("GeneratorError: Output of Generator(x) Must be a list of str")
                raise
            self.charmap,self.invcharmap = GetCharMap(self.Seqs)
            self.oh = seq2oh(self.Seqs,self.charmap)
        except:
            print("GeneratorError: Generator(x) can't run")
            raise
            
        self.Predictor = Predictor
        try:
            self.Score = np.mean(self.Predictor(self.Seqs))
            if type(self.Score) is not np.ndarray:
                print("PredictorError: Output of Predictor(Generator(x)) must be a numpy.ndarray")
                raise
            elif  len(self.Score.shape) != 1 or self.Score.shape[0] != len(self.Seqs):
                print("PredictorError: Except shape of Predictor(Generator(x)) is ({},) but got a {}".format(len(self.Seqs),str(self.Score.shape)))
                raise
        except:
            print("PredictorError: Predictor(Generator(x)) can't run")
            raise
        self.Valid = Valid
        if Valid is None:
            return
        try:
            Score = self.Valid(self.Seqs)
            if type(Score) is not np.ndarray:
                print("ValidError: Output of Valid(Generator(x)) must be a numpy.ndarray")
                raise
            elif  len(Score.shape) != 1 or Score.shape[0] != len(self.Seqs):
                print("ValidError: Except shape of Valid(Generator(x)) is ({},) but got a {}".format(len(self.Seqs),str(Score.shape)))
                raise
        except:
            print("ValidError: Valid(Generator(x)) can't run")
            raise
            
    def run(self,
            outdir='./EAresult',
            MaxPoolsize=2000,
            P_rep=0.3,
            P_new=0.25,
            P_elite=0.25,
            MaxIter=1000):
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
        if self.Valid is None:
            bestscore = np.max(self.Predictor(self.Seqs))
        else:
            bestscore = np.max(self.Valid(self.Seqs))
        for iteration in range(1,1+self.MaxIter):
            Poolsize = self.Score.shape[0]
            Nnew = math.ceil(Poolsize*self.P_new)
            Nelite = math.ceil(Poolsize*self.P_elite)
            IParent = self.select_parent( Nnew, Nelite, Poolsize) 
            Parent = self.x[IParent,:].copy()
            x_new = self.act(Parent)
            oh_new = seq2oh(self.Generator(x_new),self.charmap)
            Score_new = np.mean(self.Predictor(self.Generator(x_new)))
            self.x = np.concatenate([self.x, x_new])
            self.oh = np.concatenate([self.oh, oh_new])
            self.Score = np.append(self.Score,Score_new)
            I = np.argsort(self.Score)
            I = I[::-1]
            self.x = self.x[I,:]
            self.oh = self.oh[I,:,:]
            self.Score = self.Score[I]
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
                Seq = oh2seq(self.oh,self.invcharmap)
                np.save(outdir+'/latentv'+str(iteration),self.x)
                saveseq(outdir+'/SeqIter'+str(iteration)+'.fa',Seq)
                print('Iter {} was saved!'.format(iteration))
            
                if self.Valid is None:
                    scores.append(self.Score)
                else:
                    valscore = self.Valid(oh2seq(self.oh,self.invcharmap))
                    scores.append(valscore)
                    
                if np.max(scores[-1]) > bestscore:
                    bestscore = np.max(scores[-1])
                else:
                    break
        pdf = PdfPages(outdir+'/each_iter_distribution.pdf')
        plt.figure()
        plt.boxplot(scores)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        pdf.savefig()
        pdf.close()
        
        pdf = PdfPages(outdir+'/compared_with_natural.pdf')
        plt.figure()
        if self.Valid is None:
            nat_score = np.mean(self.Predictor(self.Seqs))
        else:
            nat_score = self.Valid(self.Seqs)
            
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
