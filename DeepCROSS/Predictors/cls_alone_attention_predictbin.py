import time
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
#from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout
from ..ops import Conv1D, Linear, ResBlock
#CNN:用keras
# from keras.layers import MaxPooling1D,UpSampling1D
# from keras.layers import Bidirectional,LSTM,Dropout,BatchNormalization
#attention:用tf
from tensorflow.layers import MaxPooling1D,Dropout,Dense
#from keras.layers import *
from ..ops.param import params_with_name
from ..ops.attention import *
import math
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from ..ProcessData import * #seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3
from sklearn.metrics import precision_score,recall_score,f1_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dropout_rate = 0.2

class CLS_alone_attention_predictbin():
    def __init__(self,Nseed,n=1,attention_num_blocks=5):
        self.nb_blocks = 2
        self.filters = 24
        #self.BATCH_SIZE =256
        self.d_model = 128
        self.c_dim = 4
        self.attention_num_blocks = 3#attention_num_blocks#3
        self.Nseed = Nseed
        self.n = n
    #########
    #1.CNN+LSTM[目前采用这个]#
    #########
    # def PredictorNet(self, x, is_training=True, reuse=False):
    #         output = Conv1D('Conv1D.1', self.c_dim, 128, 6, x)
    #         #BN1
    #         #output = BatchNormalization()(output)
    #         output = MaxPooling1D(pool_size=2)(output) 
    #         output = Conv1D('Conv1D.2', 128, 256, 3, output)
    #         #BN2
    #         #output = BatchNormalization()(output)
    #         output = MaxPooling1D(pool_size=2)(output)
    #         for i in range(1,1+8):#8个:4*8
    #             output = ResBlock(output, 256, 3, 'ResBlock.{}'.format(i))
    #         output = Conv1D('Conv_label.1', 256, 512, 3, output)
    #         #BN3
    #         #output = BatchNormalization()(output)
    #         output = Conv1D('Conv_label.2', 512, 512, 3, output)
    #         #BN4
    #         #output = BatchNormalization()(output)
    #         output = Conv1D('Conv_label.3', 512, 512, 3, output)
    #         #BN5
    #         #output = BatchNormalization()(output)
    #         output = tf.nn.relu(output)
    #         #output = tf.reshape(output, [-1, 512*int(self.SEQ_LEN/2/2)])#6.
    #         #output = Linear('Dense_label.1',512*int(self.SEQ_LEN/2/2),1024, output)#7.encoder_out
    #         #
    #         output = Dropout(0.2)(output) #保留的比例
    #         output = Bidirectional(LSTM(50,return_sequences=True))(output) #75 #(输入:bs,41,512,bs,41,100)
    #         output = Linear('Dense_label.1',41*50*2,512, output)
    #         output = Dropout(0.2)(output)
    #         #
    #         cls_logits_BS =  Linear('Dense_label.2',512,self.nbin, output)#7.
    #         cls_logits_EC =  Linear('Dense_label.3',512,self.nbin, output)#7.
    #         cls_logits_PA =  Linear('Dense_label.4',512,self.nbin, output)#7.
    #         return cls_logits_BS,cls_logits_EC,cls_logits_PA #cls_logits
    #########
    #1.#大核+attention:self-attention#
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

    def PredictorNet(self, x,is_training=True, reuse=False):
        enc = Conv1D('Conv1D.'+str(self.n), self.c_dim, self.d_model, 13, x) #（bs,165,4）-> (bs,165,512)
        enc = tf.nn.relu(enc)#tf.nn.relu(enc)
        enc = MaxPooling1D(pool_size=2,strides=2)(enc) 
        enc *= self.d_model**0.5 #scale
        enc += positional_encoding(enc, 83)#self.hp.maxlen1
        enc = Dropout(0.2)(enc)#tf.layers.dropout(enc, 0.3, training=is_training)#(bs,165,128)
        for i in range(self.attention_num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc)
                #layernorm
                enc = tf.contrib.layers.layer_norm(enc)
                # feed forward
                enc = ff(enc, num_units=[2048, self.d_model])#(bs,82,128)
        #output=enc
        output= tf.nn.relu(enc)
        output = Dropout(0.2)(output) #保留的比例
        output = Linear('Dense_rep.'+str(self.n),41*2*self.d_model,self.d_model, output)
        #output = Bidirectional(LSTM(50,return_sequences=True))(output) #75 #(输入:bs,41,512,bs,41,100)
        #output = Linear('Dense_rep',41*2*self.d_model,self.nclass, output)#
        return output

    def out_rep2y(self,last2rep):
        #output = Linear('Dense_out',self.d_model,2, last2rep)#self.nclass
        cls_logits_EC =  Linear('Dense_label.3.'+str(self.n),self.d_model,3, last2rep)#self.nbin(3)
        cls_logits_PA =  Linear('Dense_label.4.'+str(self.n),self.d_model,3, last2rep)#self.nbin(3)
        return cls_logits_EC,cls_logits_PA

    def get_cls_loss(self,EC_logits,PA_logits): #BS_logits,
        #BS=tf.expand_dims(BS_logits, 1)
        EC=tf.expand_dims(EC_logits, 1)
        PA=tf.expand_dims(PA_logits, 1)    
        logits = tf.concat((EC,PA),axis=1)
        labels=tf.cast(self.label,tf.int64)#多分类:(bs,3)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(##tf.nn.sigmoid_cross_entropy_with_logits(#
            labels=labels,
            logits=logits,
            name='cross_entropy')
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def get_cls_accuracy(self):
        labels = self.label #(128,3)#cls_predict = tf.concat((self.layers['y_BS'].expand_dims(one_img, -1),self.layers['y_EC'].expand_dims(one_img, -1),self.layers['y_PA'].expand_dims(one_img, -1)),axis=-1)#(128,) #384
        #BSresult = tf.expand_dims(self.BS_y,1)#self.layers['y_BS']
        ECresult = tf.expand_dims(self.EC_y,1)#self.layers['y_EC']
        PAresult = tf.expand_dims(self.PA_y,1)#self.layers['y_PA']
        cls_predict = tf.concat((ECresult,PAresult),axis=-1)#128,3
        cls_predict = tf.dtypes.cast(cls_predict, tf.float32)
        labels = tf.dtypes.cast(labels, tf.float32) #128,3
        #cls_predict = tf.concat((self.layers['y_BS'],self.layers['y_EC'],self.layers['y_PA']),axis=-1)#(128,) #384
        #cls_predict = tf.dtypes.cast(cls_predict, tf.float32)
        #labels = tf.reshape(tf.dtypes.cast(labels, tf.float32),[self.BATCH_SIZE*3]) #128*3=384 256*3=768
        num_correct = tf.cast(tf.equal(labels, cls_predict), tf.float32) #3 and 3
        return tf.reduce_mean(num_correct)

    def load_dataset(self,
                   train_data,
                   val_data=None,
                   log_dir='./',
                   nbin=3
                   ):
        self.nbin = nbin
        seq_exp=np.load(train_data,allow_pickle=True)
        self.charmap={'A':0,'G':1,'C':2,'T':3}
        seq_for_merge=[]
        onehot = []
        label = []
        exp_all=[]
        for i in range(len(seq_exp)):
            exp_all.append(seq_exp[i][1])
        exp_all=sum(exp_all,[])
        #1.把exp_all分为n个bin，画个图
        pdf = PdfPages(os.path.join(log_dir, 'expdistri_forbin.pdf')) 
        plt.figure(1,figsize=(7,7)) 
        n, binlist, patches=plt.hist(exp_all, bins=self.nbin, color='green',alpha=0.5, rwidth=0.85,label='PA_exp')
        pdf.savefig() 
        pdf.close()
        if self.nbin==5:
            binlist=[-20,-10,-5,0,5,10] #去掉了-15这个bin
        else:
            binlist=list(binlist)#3个bin就按照正常分即可
        # print('Expression bins:')
        # print(binlist) #
        binlist[-1]=binlist[-1]+1

        #2.固定bin:之前:#binlist=[-19.931568569324174, -10.49172410670778, -1.051879644091386, 8.387964818525008]
        #1.原始训练预测器,合成芯片:binlist=[-19.931568569324174, -10.089466579317042, -0.24736458930991034, 10.59473740069722]
        #2.MPRAfirstRound数据训练预测器:
        #binlist[0]=-2.5
        binlist=np.array(binlist) #[-1.6167405843734741, 0.022336403528849358, 1.6614133914311728, 4.300490379333496]
        print('Expression bins:')
        print(binlist) #
        exp_bin=[]
        for i in range(len(seq_exp)):
            #BS_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_BS'][i]+10**(-6))
            EC_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_EC'][i]+10**(-6))
            PA_1=seq_exp[i][1][1]#np.log2(seq_exp['tx_norm_PA'][i]+10**(-6))
            #index_BS=np.where(BS_1>=binlist)[0][-1]
            index_EC=np.where(EC_1>=binlist)[0][-1]
            #print(PA_1)
            index_PA=np.where(PA_1>=binlist)[0][-1]#[-1]
            exp_bin.append([index_EC,index_PA])#index_BS,
        for i in range(len(seq_exp)):
            seq=seq_exp[i][0]
            exp=exp_bin[i]
            #exp=seq_exp[i][1]
            seq_for_merge.append(seq)
            eachseq = np.zeros([len(seq),4],dtype = 'float')
            for j in range(len(seq)):
                base=seq[j]
                eachseq[j,self.charmap[base]] = 1
            onehot.append(eachseq)
            label.append(np.array(exp))
        self.x = np.array(onehot) #self.seq_list
        self.y = np.squeeze(label) #self.label_list (10185,3)
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
            np.random.seed(self.Nseed)
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*int(0.9*10)//10
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]#--2.[d:,:]
            self.seq_for_merge = seq_for_merge[seq_index_A[n:]]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:]#
        self.dataset_num = self.x.shape[0]
        self.dataset_num_val =self.val_x.shape[0]
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
        self.out_rep = self.PredictorNet(self.seqInput,self.training)
        self.score_EC,self.score_PA = self.out_rep2y(self.out_rep)
        #1.
        self.label = tf.placeholder(tf.float32, shape=[None,2],name='label')
        #self.BS_y = tf.argmax(self.score_BS, axis=-1,name='label_predict_BS') 
        self.EC_y = tf.argmax(self.score_EC, axis=-1,name='label_predict_EC') 
        self.PA_y = tf.argmax(self.score_PA, axis=-1,name='label_predict_PA') 
        """Loss"""
        #2.
        #self.loss = tf.losses.mean_squared_error(self.label,self.score)
        self.loss = self.get_cls_loss(self.score_EC,self.score_PA)
        self.saver = tf.train.Saver(max_to_keep=50)
        return
    
    def Train(self,
              lr=1e-4,
              beta1=0.5,
              beta2=0.9,
              epoch=1000,
              earlystop=20,
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
        self.opt = tf.train.AdamOptimizer(self.lr, beta1=beta1, beta2=beta2).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        self.cls_accuracy=self.get_cls_accuracy()

        
        counter = 1
        start_time = time.time()
        gen = self.inf_train_gen(self.dataset_num)
        gen_val = self.inf_train_gen(self.dataset_num_val)
        best_percent = 0
        convIter = 0
        #I_val = np.arange(self.dataset_num_val)
        #5.10去掉的seed:
        #np.random.seed(3)
        #np.random.shuffle(I_val)
        for epoch in range(1, 1+self.epoch):
            # get batch data
            for idx in range(1, 1+self.iteration):
                I = gen.__next__()
                _, loss,cls_accuracy_train= self.sess.run([self.opt,self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.x[I,:,:],self.label:self.y[I,:],self.training:True})
                #---3.self.label:self.y[I,:]
                # display training status
                counter += 1
                if loss <=2:
                    self.lr = 1e-6
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, loss))

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
            print("Epoch[%2d][%5d/%5d]:cls_loss_train:[%.8f],cls_loss_valid:[%.8f],cls_accuracy_train:[%.8f],cls_accuracy_valid:[%.8f]"%(epoch,idx,self.iteration,loss,loss_val,cls_accuracy_train,cls_accuracy_val))
            
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
            if cls_accuracy_val>best_percent:
                best_percent = cls_accuracy_val
                print('cls_acc best percent:'+str(best_percent)+'(epoch:'+str(epoch)+')')
                self.save(self.checkpoint_dir,epoch,'bestacc',str(best_percent))
            else:
                convIter += 1
                if convIter>=earlystop:
                    print('best acc:'+str(best_percent))
                    break
            #save
            #self.save(self.checkpoint_dir, counter)
        return

    def inf_train_gen(self,num):
        #print(' [*] Using seed:'+str(self.Nseed))
        I = np.arange(num)
        while True:
            np.random.seed(self.Nseed)
            np.random.shuffle(I)
            for i in range(0, len(I)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield I[i:i+self.BATCH_SIZE]

    def save(self, checkpoint_dir, step,name='pt',acc='0'):
        # with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
        #     for c in self.charmap:
        #         f.write(c+'\t')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir,name+'_'+str(acc)+'_'+self.model_name +'_epoch_'+str(step)+'_'+ '.model'), global_step=step)#checkpoint_dir, self.model_name + '.model'
        
    def load(self, checkpoint_dir, model_name = None):
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
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def Predictor(self,seq,datatype='str'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        num = seq.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)
        y = []
        for b in range(batches):
            y.append(self.sess.run(self.score,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
        y = np.concatenate(y)
        y = np.reshape(y,(y.shape[0]))
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
        print(' [*] Cal precision,recall and f1 ...')
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
        #写出txt
        """         
        f = open(out_dir+'/valseq_predict_expECPA.txt','w')
        for i in range(len(seq_for_merge_true)):
            f.write(seq_for_merge_true[i]+'\t'+str(EC_true[i])+'\t'+str(EC_pred[i])+'\t'+str(PA_true[i])+str(PA_pred[i])+'\n')
        f.close() 
        """
        #
        #labels = self.label #(128,3)#cls_predict = tf.concat((self.layers['y_BS'].expand_dims(one_img, -1),self.layers['y_EC'].expand_dims(one_img, -1),self.layers['y_PA'].expand_dims(one_img, -1)),axis=-1)#(128,) #384
        """         
        BS_pred = tf.dtypes.cast(tf.expand_dims(self.BS_y,1),tf.float32)#self.layers['y_BS']
        EC_pred = tf.dtypes.cast(tf.expand_dims(self.EC_y,1),tf.float32)#self.layers['y_EC']
        PA_pred = tf.dtypes.cast(tf.expand_dims(self.PA_y,1),tf.float32)#self.layers['y_PA'] 
        """
        #labels = tf.dtypes.cast(labels, tf.float32) #128,3
        #BS_true,EC_true,PA_true=tf.split(labels,num_or_size_splits=3, axis=1)
        #with tf.Session() as self.sess:
            #BS_true = BS_true.eval() #
            #BS_pred = BS_pred.eval()
        
        #macro
        # precision_macro_BS=precision_score(BS_true, BS_pred,average='macro')
        # recall_macro_BS=recall_score(BS_true, BS_pred,average='macro')
        # f1score_macro_BS=f1_score(BS_true, BS_pred,average='macro')
                
        #这里维数不同：
        precision_macro_EC=precision_score(EC_true, EC_pred,average='macro')
        recall_macro_EC=recall_score(EC_true, EC_pred,average='macro')
        f1score_macro_EC=f1_score(EC_true, EC_pred,average='macro')

        precision_macro_PA=precision_score(PA_true, PA_pred,average='macro')
        recall_macro_PA=recall_score(PA_true, PA_pred,average='macro')
        f1score_macro_PA=f1_score(PA_true, PA_pred,average='macro')

	    #micro
        # precision_micro_BS=precision_score(BS_true, BS_pred,average='micro')
        # recall_micro_BS=recall_score(BS_true, BS_pred,average='micro')
        # f1score_micro_BS=f1_score(BS_true, BS_pred,average='micro')
        
        precision_micro_EC=precision_score(EC_true, EC_pred,average='micro')
        recall_micro_EC=recall_score(EC_true, EC_pred,average='micro')
        f1score_micro_EC=f1_score(EC_true, EC_pred,average='micro')

        precision_micro_PA=precision_score(PA_true, PA_pred,average='micro')
        recall_micro_PA=recall_score(PA_true, PA_pred,average='micro')
        f1score_micro_PA=f1_score(PA_true, PA_pred,average='micro')
        #
        acc_out_macro=pd.DataFrame([[precision_macro_EC,recall_macro_EC,f1score_macro_EC],[precision_macro_PA,recall_macro_PA,f1score_macro_PA]],columns=['pr','rc','f1'],index=['EC','PA'])
        acc_out_macro.to_csv(out_dir+'/pr_rc_f1_macro_n'+str(repeatN)+'.csv')#,header=0,index=0
        #
        print(precision_micro_EC)
        print(precision_micro_PA)
        print(precision_macro_EC)
        print(precision_macro_PA)
        acc_out_micro=pd.DataFrame([[precision_micro_EC,recall_micro_EC,f1score_micro_EC],[precision_micro_PA,recall_micro_PA,f1score_micro_PA]],columns=['pr','rc','f1'],index=['EC','PA'])
        acc_out_micro.to_csv(out_dir+'/pr_rc_f1_micro_n'+str(repeatN)+'.csv')#,header=0,index=0
        return 0

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
