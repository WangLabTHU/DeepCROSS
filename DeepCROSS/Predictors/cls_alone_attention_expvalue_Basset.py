import time
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
#from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout
#from ..ops import Conv1D, Linear, ResBlock
#CNN:用keras
# from keras.layers import MaxPooling1D,UpSampling1D
# from keras.layers import Bidirectional,LSTM,Dropout,BatchNormalization
#attention:用tf
#from tensorflow.layers import MaxPooling1D,Dropout,Dense
from tensorflow.keras.layers import Conv1D,Activation,MaxPooling1D,Dense,Dropout,BatchNormalization,Flatten
from tensorflow.keras import backend as K
#from keras.layers import *
from ..ops.param import params_with_name
from ..ops.attention import *
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from scipy.stats import pearsonr,spearmanr
from ..ProcessData import * #seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3
from sklearn.metrics import precision_score,recall_score,f1_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
#from imblearn.over_sampling import SMOT

def weightMSEloss(x,y,threshold=0.4):#x=真实Label,y=预测label
    w = 2*tf.tanh(x-threshold)+3 # 2*+3#+10 #2*tf.nn.relu(x-threshold) #
    #print('w shape:')
    #print(x.shape[0])
    n= 256 #x.shape[0]
    loss = K.sum(tf.multiply(K.square(x-y),w))/n
    return loss

class CLS_alone_attention_expvalue_Basset():
    def __init__(self,Nseed,n=1,attention_num_blocks=5):
        self.nb_blocks = 2
        self.filters = 24
        self.BATCH_SIZE =256
        self.d_model = 128
        self.c_dim = 4
        self.attention_num_blocks = attention_num_blocks#3
        self.Nseed = Nseed
        self.n = n
    #########
    #1.Basset#
    #########
    def PredictorNet(self, x, is_training=True, reuse=False):
            #Conv1
            output = Conv1D(filters=128, kernel_size=7,input_shape=x.shape[1:])(x)#filters=256 246
            output = BatchNormalization()(output)
            output = Activation('relu')(output) #relu
            output = MaxPooling1D(pool_size=2)(output) 

            #Conv2
            output = Conv1D(filters=60, kernel_size=5)(output) #3,PA
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D(pool_size=2)(output)

            #Conv3
            output = Conv1D(filters=60, kernel_size=3)(output) 
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D(pool_size=2)(output)
            
            #Conv4
            output = Conv1D(filters=120, kernel_size=3)(output) 
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D(pool_size=2)(output)

            #
            #reshape
            output = Flatten()(output)
            output = Dense(256)(output)#256
            output = Dense(256)(output)#256
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = Dropout(0.2)(output) #保留的比例0.4
            #output =Dense(2)(output)
            cls_logits =  Dense(2)(output) 
            return cls_logits #cls_logits_EC,cls_logits_PA
    #########
    #2.#大核+attention:self-attention#
    #########
    # def PredictorNet0(self, x, is_training=True, reuse=False):
    #         output = Conv1D('Conv1D.1', self.c_dim, 128, 13, x) #
    #         #BN1
    #         #output = BatchNormalization()(output)
    #         output = MaxPooling1D(pool_size=2)(output) 
    #         output = Conv1D('Conv1D.2', 128, 256, 3, output)
    #         #BN2
    #         #output = BatchNormalization()(output)
    #         #1.第一层获得的embedding向量维度
            
    #         #2.queries, keys, values是相乘以后的结果：需要初始化Wq,Wk,Wv去与第一层获得的embedding向量做乘法
    #         #
    #         attn = multihead_attention(queries, keys, values)
    #         ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    #         #
    #         multi
    #         output = Dropout(0.2)(output)
    #         #
    #         cls_logits_BS =  Linear('Dense_label.2',512,self.nbin, output)#7.
    #         cls_logits_EC =  Linear('Dense_label.3',512,self.nbin, output)#7.
    #         cls_logits_PA =  Linear('Dense_label.4',512,self.nbin, output)#7.
    #         return cls_logits_BS,cls_logits_EC,cls_logits_PA #cls_logits
    #########
    #3.#大核+attention:self-attention(之前那个)#
    #########
    # def PredictorNet(self, x,is_training=True, reuse=False):
    #     enc = Conv1D('Conv1D.'+str(self.n), self.c_dim, self.d_model, 13, x) #（bs,165,4）-> (bs,165,512)
    #     enc = tf.nn.relu(enc)#tf.nn.relu(enc)
    #     enc = MaxPooling1D(pool_size=2,strides=2)(enc) 
    #     enc *= self.d_model**0.5 #scale
    #     enc += positional_encoding(enc, 83)#self.hp.maxlen1
    #     enc = Dropout(0.2)(enc)#tf.layers.dropout(enc, 0.3, training=is_training)#(bs,165,128)
    #     for i in range(self.attention_num_blocks):
    #         with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
    #             # self-attention
    #             enc = multihead_attention(queries=enc,
    #                                           keys=enc,
    #                                           values=enc)
    #             #layernorm
    #             enc = tf.contrib.layers.layer_norm(enc)
    #             # feed forward
    #             enc = ff(enc, num_units=[2048, self.d_model])#(bs,82,128)
    #     #output=enc
    #     output= tf.nn.relu(enc)
    #     output = Dropout(0.2)(output) #保留的比例
    #     output = Linear('Dense_rep.'+str(self.n),41*2*self.d_model,self.d_model, output)
    #     #output = Bidirectional(LSTM(50,return_sequences=True))(output) #75 #(输入:bs,41,512,bs,41,100)
    #     #output = Linear('Dense_rep',41*2*self.d_model,self.nclass, output)#
    #     return output

    # def out_rep2y(self,last2rep):
    #     #output = Linear('Dense_out',self.d_model,2, last2rep)#self.nclass
    #     cls_logits_EC =  Linear('Dense_label.3.'+str(self.n),self.d_model,1, last2rep)#self.nbin(3)
    #     cls_logits_PA =  Linear('Dense_label.4.'+str(self.n),self.d_model,1, last2rep)#self.nbin(3)
    #     return cls_logits_EC,cls_logits_PA

    def get_cls_loss(self,EC_logits,PA_logits): #BS_logits, #预测的y值
        labels=tf.cast(self.label,tf.float32)#int64
        EC_true,PA_true=tf.split(labels,num_or_size_splits=2, axis=1)
        #EC_y_loss=tf.losses.mean_squared_error(EC_true,EC_logits)#weightMSEloss(EC_true,EC_logits,threshold=4) #
        #PA_y_loss=tf.losses.mean_squared_error(PA_true,PA_logits)#weightMSEloss(PA_true,PA_logits,threshold=4)#
        #out=tf.add(0.3*EC_y_loss,0.7*PA_y_loss)/2
        EC_y_loss=weightMSEloss(EC_true,EC_logits,threshold=2)
        PA_y_loss=weightMSEloss(PA_true,PA_logits,threshold=0)
        out=tf.add(0.2*EC_y_loss,0.8*PA_y_loss)/2 #out=tf.add(EC_y_loss,tf.multiply(self.w,PA_y_loss))#matmul
        
        
        # #BS=tf.expand_dims(BS_logits, 1)
        # EC=tf.expand_dims(EC_logits, 1)
        # PA=tf.expand_dims(PA_logits, 1)    
        # logits = tf.concat((EC,PA),axis=1)
        # labels=tf.cast(self.label,tf.int64)#多分类:(bs,3)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(##tf.nn.sigmoid_cross_entropy_with_logits(#
        #     labels=labels,
        #     logits=logits,
        #     name='cross_entropy')
        # out = tf.reduce_mean(cross_entropy)
        return out

    def get_cls_accuracy(self):
        EC_true,PA_true=tf.split(self.label,num_or_size_splits=2, axis=1)
        EC_acc=self.pearsonr_tf(EC_true,self.EC_y)
        PA_acc = self.pearsonr_tf(PA_true,self.PA_y)
        # labels = self.label #(128,3)#cls_predict = tf.concat((self.layers['y_BS'].expand_dims(one_img, -1),self.layers['y_EC'].expand_dims(one_img, -1),self.layers['y_PA'].expand_dims(one_img, -1)),axis=-1)#(128,) #384
        # #BSresult = tf.expand_dims(self.BS_y,1)#self.layers['y_BS']
        # ECresult = tf.expand_dims(self.EC_y,1)#self.layers['y_EC']
        # PAresult = tf.expand_dims(self.PA_y,1)#self.layers['y_PA']
        # cls_predict = tf.concat((ECresult,PAresult),axis=-1)#128,3
        # cls_predict = tf.dtypes.cast(cls_predict, tf.float32)
        # labels = tf.dtypes.cast(labels, tf.float32) #128,3
        # #cls_predict = tf.concat((self.layers['y_BS'],self.layers['y_EC'],self.layers['y_PA']),axis=-1)#(128,) #384
        # #cls_predict = tf.dtypes.cast(cls_predict, tf.float32)
        # #labels = tf.reshape(tf.dtypes.cast(labels, tf.float32),[self.BATCH_SIZE*3]) #128*3=384 256*3=768
        # num_correct = tf.cast(tf.equal(labels, cls_predict), tf.float32) #3 and 3
        # out = tf.reduce_mean(num_correct)
        return [EC_acc,PA_acc] #out

    def load_dataset(self,
                   train_data,
                   val_data=None,
                   log_dir='./',
                   ):
        seq_exp=np.load(train_data,allow_pickle=True)
        self.charmap={'A':0,'G':1,'C':2,'T':3}
        seq_for_merge=[]
        onehot = []
        label = []
        exp_all=[]
        #for i in range(len(seq_exp)):
            #exp_all.append(seq_exp[i][1])
        # exp_all=sum(exp_all,[])
        # #1.把exp_all分为n个bin，画个图
        # pdf = PdfPages(os.path.join(log_dir, 'expdistri_forbin.pdf')) 
        # plt.figure(1,figsize=(7,7)) 
        # n, binlist, patches=plt.hist(exp_all, bins=self.nbin, color='green',alpha=0.5, rwidth=0.85,label='PA_exp')
        # pdf.savefig() 
        # pdf.close()
        # if self.nbin==5:
        #     binlist=[-20,-10,-5,0,5,10] #去掉了-15这个bin
        # else:
        #     binlist=list(binlist)#3个bin就按照正常分即可
        # # print('Expression bins:')
        # # print(binlist) #
        # binlist[-1]=binlist[-1]+1

        # #2.固定bin:之前:#binlist=[-19.931568569324174, -10.49172410670778, -1.051879644091386, 8.387964818525008]
        # #1.原始训练预测器,合成芯片:binlist=[-19.931568569324174, -10.089466579317042, -0.24736458930991034, 10.59473740069722]
        # #2.MPRAfirstRound数据训练预测器:
        # #binlist[0]=-2.5
        # binlist=np.array(binlist) #[-1.6167405843734741, 0.022336403528849358, 1.6614133914311728, 4.300490379333496]
        # print('Expression bins:')
        # print(binlist) #
        # exp_bin=[]
        # for i in range(len(seq_exp)):
        #     #BS_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_BS'][i]+10**(-6))
        #     EC_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_EC'][i]+10**(-6))
        #     PA_1=seq_exp[i][1][1]#np.log2(seq_exp['tx_norm_PA'][i]+10**(-6))
        #     #index_BS=np.where(BS_1>=binlist)[0][-1]
        #     index_EC=np.where(EC_1>=binlist)[0][-1]
        #     #print(PA_1)
        #     index_PA=np.where(PA_1>=binlist)[0][-1]#[-1]
        #     exp_bin.append([index_EC,index_PA])#index_BS,
        EC_norm=[]
        PA_norm=[]
        seqs=[]
        EC_repeat =[]
        PA_repeat=[]
        print('filter -2:') #-3
        print(np.array(seq_exp[0][1][0]))
        print(seq_exp[0][1][0])
        Nresample=15
        for i in range(len(seq_exp)):
            if list(np.array(seq_exp[i][1]))[0] >-2 and list(np.array(seq_exp[i][1]))[0]<4 and list(np.array(seq_exp[i][1]))[1] >-2 and list(np.array(seq_exp[i][1]))[1] <2: #过滤掉PA小于-2的那些
                #组内
                EC_norm.append(np.array(seq_exp[i][1][0]))
                PA_norm.append(np.array(seq_exp[i][1][1]))
                seqs.append(seq_exp[i][0])
                # if list(np.array(seq_exp[i][1]))[0] >= 3 or list(np.array(seq_exp[i][1]))[1] >= 0.6:
                #     EC_norm.extend([seq_exp[i][1][0]]*Nresample)
                #     PA_norm.extend([seq_exp[i][1][1]]*Nresample)
                #     seqs.extend([seq_exp[i][0]]*Nresample)
                # else:
                #     EC_norm.extend([seq_exp[i][1][0]])
                #     PA_norm.extend([seq_exp[i][1][1]])
                #     seqs.extend([seq_exp[i][0]])

        #norm:
        #EC_norm = preprocessing.scale(EC_norm)
        #PA_norm = preprocessing.scale(PA_norm)
        EC_norm2 = np.array(EC_norm)
        PA_norm2 = np.array(PA_norm)


        
        print('after filter: EC and PA')
        print(np.min(EC_norm2))
        print(np.min(PA_norm2))

        for i in range(len(seqs)):#seq_exp
            #if list(np.array(seq_exp[i][1]))[0] >-4 and list(np.array(seq_exp[i][1]))[1] >-4: #过滤掉PA小于0的那些
            seq=seqs[i]#seq_exp[i][0]
            exp=[EC_norm[i],PA_norm[i]]#list(np.array(seq_exp[i][1])) 

        # for i in range(len(seq_exp)):
        #     #if list(np.array(seq_exp[i][1]))[1] >0: #过滤掉PA小于0的那些
        #     seq=seq_exp[i][0]
        #     exp=list(np.array(seq_exp[i][1])) #+2
            seq_for_merge.append(seq)
            eachseq = np.zeros([len(seq),4],dtype = 'float')
            for j in range(len(seq)):
                base=seq[j]
                eachseq[j,self.charmap[base]] = 1
            onehot.append(eachseq)
            label.append(np.array(exp))
            #label=np.array(exp)
        self.x = np.array(onehot) #self.seq_list
        self.y = np.squeeze(label) #self.label_list (10185,3)
        print('after filter: y')
        print(np.min(self.y))
        seq_for_merge = np.array(seq_for_merge)
        print('seq_for_merge shape:')
        print(seq_for_merge.shape)
        print(np.unique(self.y,axis=0))
        print(len(np.unique(self.y,axis=0))) #27
        print(self.y.shape)
        #self.x,self.y = load_fun_data_exp3(train_data,flag=1,already_log=True)
        #self.charmap, self.invcharmap = GetCharMap(self.x)
        #self.x = seq2oh(self.x,self.charmap)
        self.y=np.reshape(self.y,(self.y.shape[0],2))
        self.SEQ_LEN = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            #d = self.x.shape[0]//10 *9
            #5.10去掉的
            np.random.seed(21)#self.Nseed
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*int(0.8*10)//10 #
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]#--2.[d:,:]
            self.seq_for_merge = seq_for_merge[seq_index_A[n:]]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:]#
            #重采样
            #sample_solver = SMOTE(random_state=0)
            #X_sample ,y_sample = sample_solver.fit_sample(self.x,self.y)
            self.x_new=[]
            self.y_new=[]
            for i in range(len(self.y)):
                if self.y[i][0] >= 3 or self.y[i][1] >= 0.6:
                        self.y_new.extend([self.y[i][0],self.y[i][1]]*Nresample)
                        self.x_new.extend([self.x[i]]*Nresample)
                else:
                        self.y_new.extend([self.y[i][0],self.y[i][1]])
                        self.x_new.extend([self.x[i]])
            self.x = np.array(self.x_new)
            self.y = np.array(self.y_new)
            self.y = self.y.reshape(-1,2)
            print('resample result:')
            print(self.x.shape)
            print(self.y.shape)

        self.dataset_num = self.x.shape[0]
        self.dataset_num_val =self.val_x.shape[0]
        #输入的表达量分布
        pdf = PdfPages(os.path.join(log_dir,'Input_exp_distribution_y.pdf'))
        plt.figure(1)
        sns.distplot(self.y)
        plt.xlabel('Input_exp_distribution_y')
        pdf.savefig()
        pdf.close()
        #
        pdf = PdfPages(os.path.join(log_dir,'Input_exp_distribution_valy.pdf'))
        plt.figure(2)
        sns.distplot(self.val_y)
        plt.xlabel('Input_exp_distribution_valy')
        pdf.savefig()
        pdf.close() 


        data_label = pd.DataFrame([EC_norm,PA_norm]).T
        data_label.columns=['EC','PA']
        #EC和PA的相关性
        pdf = PdfPages(os.path.join(log_dir,'EC_PA_cor.pdf'))
        plt.figure(10,figsize=(10, 10))
        sns.scatterplot(x='EC', y='PA', data=data_label,color='#000000',marker='.')
        plt.xlabel('EC_MPRAmy')
        plt.ylabel('PA_MPRAmy')
        plt.xlim(-2,4)
        plt.ylim(-2,4)
        pdf.savefig()
        pdf.close()
        return

    def Encoder_rep_beforelastlinear(self,seq,datatype='str'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        out_rep=self.sess.run(self.out_rep,feed_dict={self.seqInput:seq})#
        return out_rep  
        
    def BuildModel(self,
                   DIM = 256,
                   kernel_size = 3
                   ):
        tf.reset_default_graph()
        self.DIM = DIM
        self.kernel_size = kernel_size
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
         #config=tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7   # 最大显存占用率70%
        # config.allow_soft_placement = True 

        """Model"""
        self.training = tf.placeholder(tf.bool)#
        self.seqInput = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
        self.score = self.PredictorNet(self.seqInput,self.training) #self.score_EC,self.score_PA 
        #1.
        self.label = tf.placeholder(tf.float32, shape=[None,2],name='label')
        #self.BS_y = tf.argmax(self.score_BS, axis=-1,name='label_predict_BS') 
        self.EC_y = tf.split(self.score,2,axis=1)[0] #tf.argmax(self.score_EC, axis=-1,name='label_predict_EC') 
        self.PA_y = tf.split(self.score,2,axis=1)[1] #tf.argmax(self.score_PA, axis=-1,name='label_predict_PA') 
        self.w = tf.placeholder(tf.float32,shape=[None],name='w')

        """Loss"""
        #2.
        #self.loss = tf.losses.mean_squared_error(self.label,self.score)
        self.loss = self.get_cls_loss(self.EC_y,self.PA_y)
        self.saver = tf.train.Saver(max_to_keep=50)
        return
    
    def Train(self,
              lr=1e-4,
              beta1=0.5,
              beta2=0.9,
              epoch=1000,
              earlystop=10,
              batch_size=32,
              checkpoint_dir='./predict_model',
              model_name='cls_alone'
              ):
        self.lr=lr
        self.BATCH_SIZE = batch_size
        self.epoch = epoch
        self.iteration = self.dataset_num // self.BATCH_SIZE
        self.earlystop = earlystop
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name

        self.global_step = tf.placeholder(tf.float32, shape=())
        self.lr = tf.train.exponential_decay(0.002, self.global_step,5,0.96,staircase=True)#5个epoch

        self.opt = tf.train.AdamOptimizer(self.lr, beta1=beta1, beta2=beta2).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        self.cls_accuracy=self.get_cls_accuracy()

        
        counter = 1
        start_time = time.time()
        gen = self.inf_train_gen(self.dataset_num)
        gen_val = self.inf_train_gen(self.dataset_num_val)
        best_loss = 100000
        best_percent=0
        convIter = 0
        #I_val = np.arange(self.dataset_num_val)
        #5.10去掉的seed:
        #np.random.seed(3)
        #np.random.shuffle(I_val)
        for epoch in range(1, 1+self.epoch):
            # get batch data
            for idx in range(1, 1+self.iteration):
                I = gen.__next__()
                _,lr,loss,cls_accuracy_train= self.sess.run([self.opt,self.lr,self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.x[I,:,:],self.label:self.y[I,:],self.training:True,self.global_step:epoch})#,
                #---3.self.label:self.y[I,:]
                # display training status
                counter += 1
                # if loss <=2:
                #     self.lr = 1e-6 #,lr:%.8f
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f,lr:%.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, loss,lr))

            # train_pred = self.Predictor(self.x,'oh')
            # train_pred = np.reshape(train_pred,(train_pred.shape[0],1))
            # train_R = pearsonr(train_pred,self.y)[0]
            # val_pred = self.Predictor(self.val_x,'oh')
            # val_pred = np.reshape(val_pred,(val_pred.shape[0],1))
            # val_R = pearsonr(val_pred,self.val_y)[0]
            # print('Epoch {}: train R: {}, val R: {}'.format(
            #         epoch,
            #         train_R,
            #         val_R))
            #loss_val,cls_accuracy_val = self.sess.run([self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.val_x[I_val,:,:],self.label:self.val_y[I_val,:],self.training:False})
            I_val = gen_val.__next__()
            loss_val,cls_accuracy_val = self.sess.run([self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.val_x[I_val,:,:],self.label:self.val_y[I_val,:],self.training:False})
            #loss_val,cls_accuracy_val = self.sess.run([self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.val_x,self.label:self.val_y,self.training:False})
            cls_accuracy_val_EC=cls_accuracy_val[0]
            cls_accuracy_val_PA=cls_accuracy_val[1]
            cls_accuracy_train_EC = cls_accuracy_train[0]
            cls_accuracy_train_PA = cls_accuracy_train[1]
            print("Epoch[%2d][%5d/%5d]:cls_loss_train:[%.8f],cls_loss_valid:[%.8f],cls_accuracy_train_EC:[%.8f],cls_accuracy_train_PA:[%.8f],cls_accuracy_valid_EC:[%.8f],cls_accuracy_valid_PA:[%.8f]"%(epoch,idx,self.iteration,loss,loss_val,cls_accuracy_train_EC,cls_accuracy_train_PA,cls_accuracy_val_EC,cls_accuracy_val_PA))
            
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

            # save model
            # if cls_accuracy_val>best_percent:
            #     best_percent = cls_accuracy_val
            #     self.save(self.checkpoint_dir, counter)
            # else:
            #     convIter += 1
            #     if convIter>=earlystop:
            #         break
            # if cls_accuracy_val>best_percent:
            #     best_percent = cls_accuracy_val
            #     print('cls_acc best percent:'+str(best_percent)+'(epoch:'+str(epoch)+')')
            #     self.save(self.checkpoint_dir,epoch,'bestacc',str(best_percent))
            # else:
            #     convIter += 1
            #     if convIter>=earlystop:
            #         print('best acc:'+str(best_percent))
            #         break
            
            if loss_val < best_loss: #np.sum(cls_accuracy_val) > np.sum(best_percent): #loss_val
                best_loss = loss_val
                best_percent = cls_accuracy_val
                print('cls_acc loss:'+str(best_loss)+'(epoch:'+str(epoch)+')')
                self.save(self.checkpoint_dir,epoch,'bestloss',cls_accuracy_val)
            else:
                convIter += 1
                if convIter>=earlystop:
                    print('best acc(EC):'+str(best_percent[0])+'best acc(PA):'+str(best_percent[1]))
                    break
            #save
            #self.save(self.checkpoint_dir, counter)

        ######原本在这里
        return

    def inf_train_gen(self,num):
        #print(' [*] Using seed:'+str(self.Nseed))
        I = np.arange(num)
        while True:
            np.random.seed(self.Nseed)
            np.random.shuffle(I)
            for i in range(0, len(I)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield I[i:i+self.BATCH_SIZE]

    def save(self, checkpoint_dir, step,name='pt',acc=[0,0]):
        # with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
        #     for c in self.charmap:
        #         f.write(c+'\t')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir,name+'_ECacc_'+str(acc[0])+'_PAacc_'+str(acc[1])+'_'+self.model_name +'_epoch_'+str(step)+'_'+ '.model'), global_step=step)#checkpoint_dir, self.model_name + '.model'
        


    def load(self, checkpoint_dir= None, model_name = None,log_dir=None):
        print(" [*] Reading checkpoints...")
        if checkpoint_dir == None:
            checkpoint_dir = self.checkpoint_dir
        if model_name == None:
            model_name = self.model_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            #
            train_pred = self.Predictor(self.x,'oh')
            val_pred = self.Predictor(self.val_x,'oh')
            #EC:0;PA:1
            plot_ECorPA_list=['EC','PA']
            color_list=['red','blue']
            xlimECPA = [4,2]
            #datadjx = pd.DataFrame({'xdjx': [-2,5], 'ydjx': [-2,5]})#-3,9
            for plot_ECorPA in range(len(plot_ECorPA_list)):
                print('plot_ECorPA:')
                print(plot_ECorPA)
                spe = plot_ECorPA_list[plot_ECorPA]
                #Val:
                data_val = pd.DataFrame(columns=['Promoter activity by MPRA','Predicted promoter activity'])
                data_val['Predicted promoter activity']=list(np.split(val_pred,2,axis=1)[plot_ECorPA].reshape(-1,)) #np.split(y,2,axis=1)[0]
                data_val['Promoter activity by MPRA']=list(np.split(self.val_y,2,axis=1)[plot_ECorPA].reshape(-1,))
                val_R_PCC = pearsonr(data_val['Promoter activity by MPRA'],data_val['Predicted promoter activity'])[0]
                val_R_spearman = spearmanr(data_val['Promoter activity by MPRA'],data_val['Predicted promoter activity'])[0]
                print(spe+': val_R_PCC='+str(val_R_PCC))
                print(spe+': val_R_spearman='+str(val_R_spearman))
                pdf = PdfPages(os.path.join(log_dir,'true_pred_scatter_val_'+spe+'.pdf'))
                plt.figure(int(3+plot_ECorPA),figsize=(10, 10))
                sns.set(style="whitegrid",font_scale=1.2)
                g=sns.regplot(x='Promoter activity by MPRA', y='Predicted promoter activity', data=data_val,
                    color='#000000',
                    marker='.',
                    scatter_kws={'s': 5,'color':color_list[plot_ECorPA]},#设置散点属性，参考plt.scatter
                    line_kws={'linestyle':'--','color':'darkgrey'})#设置线属性，参考 plt.plot 
                #g2=sns.lineplot(data=datadjx, x="xdjx", y="ydjx",dashes=True)
                plt.xlabel('Promoter activity by MPRA')
                plt.ylabel('Predicted promoter activity(val)')
                plt.xlim(-1,xlimECPA[plot_ECorPA]) #EC:-1,4; PA:-1,2
                plt.ylim(-1,xlimECPA[plot_ECorPA])
                pdf.savefig()
                pdf.close()

                #Train:
                data_train = pd.DataFrame(columns=['Promoter activity by MPRA','Predicted promoter activity'])
                data_train['Promoter activity by MPRA'] = list(np.split(self.y,2,axis=1)[plot_ECorPA].reshape(-1,))
                data_train['Predicted promoter activity']=list(np.split(train_pred,2,axis=1)[plot_ECorPA].reshape(-1,))
                train_R = pearsonr(data_train['Promoter activity by MPRA'],data_train['Predicted promoter activity'])[0]
                print(spe+': train_R='+str(train_R))
                pdf = PdfPages(os.path.join(log_dir,'true_pred_scatter_train_'+spe+'.pdf'))
                plt.figure(int(5+plot_ECorPA),figsize=(10, 10))
                g=sns.regplot(x='Promoter activity by MPRA', y='Predicted promoter activity', data=data_train,
                    color='#000000',
                    marker='+',
                    scatter_kws={'s': 40,'color':'grey',},#设置散点属性，参考plt.scatter
                    line_kws={'linestyle':'--','color':'r'})#设置线属性，参考 plt.plot 
                plt.xlabel('Promoter activity by MPRA')
                plt.ylabel('Predicted promoter activity(train)')
                pdf.savefig()
                pdf.close()
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def Predictor(self,seq,datatype='str',spe='ECandPA'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        num = seq.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)
        y = []
        if spe =='EC':
            for b in range(batches):
                y.append(self.sess.run(self.EC_y,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
            y = np.concatenate(y)
            y = np.reshape(y,(y.shape[0]))
        elif spe =='PA':
            for b in range(batches):
                y.append(self.sess.run(self.PA_y,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
            y = np.concatenate(y)
            y = np.reshape(y,(y.shape[0]))
        else:
            for b in range(batches):
                eachy_final=[]
                eachy=self.sess.run([self.EC_y,self.PA_y],feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]})
                eachy=np.array(eachy) #2,256,1
                eachy = np.squeeze(eachy)#2,256 2,251
                for i in range(len(eachy[0])):
                    eachy_final.append([eachy[0][i],eachy[1][i]])
                eachy_final = np.array(eachy_final)
                y.append(eachy_final)
            y = np.concatenate(y)
        return y

    def get_pred_ture(self,gen_val):
        EC_true=[]
        PA_true=[]
        EC_pred=[]
        PA_pred=[]
        seq_for_merge_true=[]
        I_val = gen_val.__next__()
        EC_pred_each,PA_pred_each= self.sess.run([self.EC_y,self.PA_y],feed_dict={self.seqInput:self.val_x[I_val,:,:]})#self.BS_y,self.EC_y,self.PA_
        label_true=self.val_y[I_val,:]
        seq_for_merge_true_each = self.seq_for_merge[I_val]
        EC_true_each,PA_true_each = np.hsplit(label_true,2) 
        #
        EC_true.append(EC_true_each)
        PA_true.append(PA_true_each)
        EC_pred.append(EC_pred_each)
        PA_pred.append(PA_pred_each)
        seq_for_merge_true.append(seq_for_merge_true_each)
        return  EC_pred,PA_pred,EC_true,PA_true,seq_for_merge_true #(256,1): 数字,不是onehot标签

    def Evalu_acc(self,repeatN,out_dir):
        print(' [*] Cal acc ...')
        gen_val = self.inf_train_gen(self.dataset_num_val)
        EC_pred,PA_pred,EC_true,PA_true,seq_for_merge_true=self.get_pred_ture(gen_val)
        EC_true = list(np.squeeze(EC_true))
        PA_true = list(np.squeeze(PA_true))
        EC_pred = list(np.squeeze(EC_pred))#np.array
        PA_pred = list(np.squeeze(PA_pred))#np.array
        seq_for_merge_true = list(np.array(seq_for_merge_true))
        print('EC_true:')
        print(np.array(EC_true).shape) #16?
        print('EC_pred:')
        print(np.array(EC_pred).shape) #1?
        #这里维数不同：
        EC_r = self.pearsonr(EC_true,EC_pred) 
        PA_r = self.pearsonr(PA_true,PA_pred) 
        print('PCC-EC:'+str(EC_r))
        print('PCC-PA:'+str(PA_r))
        #
        #self.load()
        return 0

    def pearsonr(self,x,y):
        mx = np.mean(x)
        my = np.mean(y)
        xm, ym = x-mx, y-my
        r_num = np.sum(xm * ym)
        r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
        r = r_num / r_den
        return r

    def pearsonr_tf(self,x,y):
        mx = tf.reduce_mean(x)
        my = tf.reduce_mean(y)
        xm, ym = x-mx, y-my
        r_num = tf.reduce_sum(xm * ym)
        r_den = tf.sqrt(tf.reduce_sum(xm*xm) * tf.reduce_sum(ym*ym))
        r = r_num / r_den
        return r

    # def pearsonr_tf(self,x,y):
    #     mx = tf.reduce_mean(x, axis=1, keepdims=True)
    #     my = tf.reduce_mean(y, axis=1, keepdims=True)
    #     xm, ym = x - mx, y - my
    #     t1_norm = tf.nn.l2_normalize(xm, axis = 1)
    #     t2_norm = tf.nn.l2_normalize(ym, axis = 1)
    #     cosine = tf.losses.cosine_distance(t1_norm, t2_norm, axis = 1)
    #     return cosine

    # def pearsonr_tf(self,x,y):
    #     out,_ =tf.contrib.metrics.streaming_pearson_correlation(x,y)
    #     return out

    ############
    #result avg
    ############
    def get_pred_exp(self):
        print(' [*] get pred_exp ...')
        gen_val = self.inf_train_gen(self.dataset_num_val)
        EC_pred,PA_pred,EC_true,PA_true,seq_for_merge_true=self.get_pred_ture(gen_val)
        EC_true = list(np.squeeze(EC_true))
        PA_true = list(np.squeeze(PA_true))
        EC_pred = list(np.squeeze(EC_pred))#np.array
        PA_pred = list(np.squeeze(PA_pred))#np.array
        #seq_for_merge_true = list(np.array(seq_for_merge_true))
        print('EC_true:')
        print(np.array(EC_true).shape) #16?
        print('EC_pred:')
        print(np.array(EC_pred).shape) #1?
        #
        pred_exp_eachrepeat = [EC_pred,PA_pred]
        true_exp_eachrepeat = [EC_true,PA_true]
        return  pred_exp_eachrepeat,true_exp_eachrepeat





    def get_z(self,rep_dir,name_list_3,name_list_7,gen_bs,explabelflag=0,gen_num=3000):
        rep_data_7=[]
        speflag_7=[]
        #同一个物种的序列
        for k in range(len(name_list_7)):
            fname=name_list_7[k]
            print(fname)
            with open(fname+'.pickle','rb') as f: #self.log_dir+'/'+
                    promoter=pickle.load(f)
            with open(fname+'_label.pickle','rb') as f: #self.log_dir+'/'+
                    label_in=pickle.load(f)
            print(len(promoter))
            print(label_in[0])
            if len(promoter)>=256: 
                gen_bs=256
            elif len(promoter)>=32: 
                gen_bs=32
            else:
                gen_bs=7
            promoter = seq2oh(promoter,self.charmap)
            for j in range(int(len(promoter)/gen_bs)):
                input_oh= promoter[j*gen_bs:(j+1)*gen_bs,:,:]
                #input_label= label_in[j*gen_bs:(j+1)*gen_bs]
                #each_data_z = self.Encoder_z(input_oh,datatype='oh')
                each_data = self.Encoder_rep_beforelastlinear(input_oh,datatype='oh')#self.Generate_rep(input_label,each_data_z,gen_bs)#self.sess.run(self.rep,feed_dict={self.z:each_data_z})
                #
                #print(np.array(each_data).shape)
                rep_data_7.append(each_data)
                speflag_7.append([fname]*each_data.shape[0]) 
        #
        # with open(self.log_dir+'/AAE_generated_rep', 'rb') as f:#self.log_dir+'/'+
        #     self.artifical_rep=pickle.load(f)
        # with open(self.log_dir+'/AAE_generated_label', 'rb') as f:
        #     self.artifical_spe=pickle.load(f)
        # print(np.array(self.artifical_rep).shape) #(2304,66)
        # print(np.array(rep_data_7).shape) #(29,)
        # print(np.array(speflag_7).shape)#(29,)

        # rep_data_7.append(self.artifical_rep)
        # speflag_7.append(self.artifical_spe)
        #
        rep_data_7=np.concatenate(rep_data_7) #np.array(rep_data)#
        speflag_7=np.concatenate(speflag_7)
        print('latent data_7 shape:')
        print(rep_data_7.shape)
        #save
        rep_data_file_7='rep_data_2kevolurep_7.pickle'
        speflag_file_7='speflag_2kevolurep_7.pickle'
        with open(rep_dir+rep_data_file_7, 'wb') as f:
                pickle.dump(rep_data_7, f) 
        with open(rep_dir+speflag_file_7, 'wb') as f:
                pickle.dump(speflag_7, f) 
        print('saved!')
        #
        #
        #label_3
        if explabelflag:
            rep_data_3=[]
            speflag_3=[]
            exp_value_3=[]
            for k in range(len(name_list_3)):
                fname=name_list_3[k]
                print(fname)
                with open(fname+'.pickle','rb') as f: #self.log_dir+'/'+
                        promoter=pickle.load(f)
                with open(fname+'_label.pickle','rb') as f: #self.log_dir+'/'+
                        label_in=pickle.load(f)
                with open(fname+'_exp.pickle','rb') as f: #self.log_dir+'/'+
                        exp_in=pickle.load(f)
                print(len(promoter))
                print(label_in[0])
                print(exp_in[0])
                if len(promoter)>=256: 
                    gen_bs=256
                elif len(promoter)>=32: 
                    gen_bs=32
                else:
                    gen_bs=7
                promoter = seq2oh(promoter,self.charmap)
                for j in range(int(len(promoter)/gen_bs)):
                    input_oh= promoter[j*gen_bs:(j+1)*gen_bs,:,:]
                    input_label= label_in[j*gen_bs:(j+1)*gen_bs]
                    #each_data_z= self.sess.run(self.gen_z,feed_dict={self.real_input:input_oh})
                    #
                    each_data_z = self.Encoder_z(input_oh,datatype='oh')
                    each_data = self.Generate_rep(input_label,each_data_z,gen_bs)#self.sess.run(self.rep,feed_dict={self.z:each_data_z})
                    #
                    rep_data_3.append(each_data)
                    speflag_3.append([fname]*each_data.shape[0]) 
                    exp_value_3.append(exp_in[j*gen_bs:(j+1)*gen_bs])
            rep_data_3=np.concatenate(rep_data_3) #np.array(rep_data)#
            speflag_3=np.concatenate(speflag_3)
            exp_value_3=np.concatenate(exp_value_3)
            print('latent data_3 shape:')
            print(rep_data_3.shape)
            #save
            rep_data_file_3='rep_data_EC_BS_PA_3.pickle'
            speflag_file_3='speflag_EC_BS_PA_3.pickle'
            exp_value_file_3='expvalue_EC_BS_PA_3.pickle'
            with open(rep_dir+rep_data_file_3, 'wb') as f:
                    pickle.dump(rep_data_3, f) 
            with open(rep_dir+speflag_file_3, 'wb') as f:
                    pickle.dump(speflag_3, f) 
            with open(rep_dir+exp_value_file_3, 'wb') as f:
                    pickle.dump(exp_value_3, f) 
            print('saved!')#
            return rep_data_3,rep_data_7,speflag_3,speflag_7,exp_value_3
        else:
            return rep_data_7,speflag_7
