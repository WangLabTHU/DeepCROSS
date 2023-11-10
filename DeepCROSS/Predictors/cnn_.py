import time
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout
from ..ops import Conv1D as conv1d
from ..ops import Linear as linear
from ..ops import ResBlock
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D
from keras.layers import *
from keras import regularizers
from ..ops.param import params_with_name
import math
import numpy as np
import os
from scipy.stats import spearmanr
from ..ProcessData import seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3,load_fun_data_npy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns


def pearsonr(x,y):
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    r = r_num / r_den
    return r

class CNN:
    def load_dataset(self,log_dir,train_data,train_test_ratio=0.9,val_data=None,singleflag=False):
        if singleflag:
            self.x,self.y = load_fun_data_npy(train_data)
        else:
            self.x,self.y = load_fun_data_exp3(train_data,flag=1,already_log=True)
        #self.y = (self.y-min(self.y))/(max(self.y)-min(self.y))
        self.y = np.reshape(self.y,(self.y.shape[0],1))
        self.charmap, self.invcharmap = GetCharMap(self.x)
        self.x = seq2oh(self.x,self.charmap)
        self.seq_len = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            np.random.seed(21)
            seq_index =  np.arange(self.x.shape[0])
            np.random.shuffle(seq_index)
            d = self.x.shape[0]*int(train_test_ratio*10)//10
            #train_index=seq_index[:n]
            #d = self.x.shape[0]//10 *9
            self.val_x, self.val_y = self.x[seq_index[d:],:,:], self.y[seq_index[d:],:]
            self.x, self.y = self.x[seq_index[:d],:,:], self.y[seq_index[:d],:]#,:
        self.dataset_num = self.x.shape[0]
        #输入的表达量分布
        pdf = PdfPages(log_dir+'/Input_exp_distribution_train_val_y.pdf')
        sns.distplot(self.y)
        sns.distplot(self.val_y)
        plt.legend()
        plt.xlabel('Input_exp_distribution')
        pdf.savefig()
        pdf.close()
        #
        # pdf = PdfPages(log_dir+'/Input_exp_distribution_val_y.pdf')
        # sns.distplot(self.val_y)
        # plt.xlabel('Input_exp_distribution')
        # pdf.savefig()
        # pdf.close()
        return 

    #相比于model3,差一点
    """     
    def PredictorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):
            output = conv1d('conv1d.1', self.c_dim, self.DIM, 1, x)
            output = tf.reshape(output, [-1, self.seq_len, self.DIM])
            output = MaxPooling1D(pool_size=2)(output)
            for i in range(1,2):
                output = ResBlock(output, self.DIM, self.kernel_size, 'resblock.{}'.format(i))
            output = tf.reshape(output, [-1, int(self.seq_len/2)*self.DIM])
            output = linear('y', int(self.seq_len/2)*self.DIM, 1, output)
            return output   
    """

        
    #LSTM+chx:不行,train也不行
    #"""     
    def PredictorNet(self,x,is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):
            regW = regularizers.l2(0.00001)
            #seqInput = Input(shape=(self.seq_len,4),name='seqInput') #Maxlength
            x = Conv1D(filters=128,kernel_size=6,padding="valid",activation="relu")(x)#64
            # x = Conv1D(filters=64,kernel_size=3,padding="valid",activation="relu")(x)
            # x = Conv1D(filters=64,kernel_size=3,padding="valid",activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(filters=128*2,kernel_size=3,padding="valid",activation="relu")(x)
            #x = Conv1D(filters=64,kernel_size=3,padding="valid",activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            #x = Conv1D(filters=64,kernel_size=3,padding="valid",activation="relu")(x)
            x = Conv1D(filters=128*4,kernel_size=3,padding="valid",activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            #现在是22,应该再加2个conv+Maxpooling
            x = Dropout(0.5)(x) #0.2,保留的比例
            x = Bidirectional(LSTM(75,return_sequences=True))(x) #75
            x = Dropout(0.5)(x)
            x = Flatten()(x)
            x = Dense(32,activation='relu')(x) #512
            x = Dropout(0.5)(x)
            y = Dense(1,kernel_regularizer= regW)(x) #kernel_regularizer= regW
        return y #Model(inputs=[seqInput],outputs=[y])  
    #"""

    """  200bp:   
    def PredictorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):
            regW = regularizers.l2(0.001)
            x = Conv1D(self.DIM, self.kernel_size, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM, 3, activation='relu')(x)#self.DIM*2,self.kernel_size
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM, 3, activation='relu')(x)#self.DIM*4,self.kernel_size
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x=Flatten()(x)
            x = Dropout(0.5)(x)#保留0.5
            y = Dense(1,kernel_regularizer= regW)(x)
            return y    
    """
    
    """     
    def PredictorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):
            #50bp的感受野
            regW = regularizers.l2(0.001)
            x = Conv1D(self.DIM, 13, activation='relu')(x)#6 #strides=2, #self.kernel_size
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            #x = Dropout(0.5)(x)
            x = Conv1D(self.DIM*2, 3, activation='relu')(x)##DIM*2 1.5  #self.DIM*2,self.kernel_size
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)#增加dropout

            # x = Conv1D(self.DIM*4, 3, activation='relu')(x)#self.DIM*4,self.kernel_size
            # x = MaxPooling1D(pool_size=2)(x)
            # x = BatchNormalization()(x)

            # #x = Dropout(0.5)(x)
            # x = Conv1D(self.DIM*8, 3, activation='relu')(x)#self.DIM*4,self.kernel_size
            # x = MaxPooling1D(pool_size=2)(x)
            # x = BatchNormalization()(x)
            # ##200再加感受野(加完后感受野有100bp左右)            
            # x = Conv1D(self.DIM*8, 3, activation='relu')(x)#self.DIM*4,self.kernel_size
            # x = MaxPooling1D(pool_size=2)(x)
            # x = BatchNormalization()(x)
            # ##200再加感受野(加完后感受野有200bp左右)         
            # x = Conv1D(self.DIM*8, 3, activation='relu')(x)#self.DIM*4,self.kernel_size
            # x = MaxPooling1D(pool_size=2)(x)
            # x = BatchNormalization()(x)
            # #
            x=Flatten()(x)
            x = Dropout(0.5)(x)#保留0.5
            y = Dense(1,kernel_regularizer= regW)(x) #2.kernel_regularizer= regW
            return y   
    """

    def BuildModel(self,
                   DIM = 128,
                   kernel_size = 5,
                   batch_size=32,
                   model_name='cnn'
                   ):
        self.DIM = DIM
        self.kernel_size = kernel_size
        self.BATCH_SIZE = batch_size
        self.model_name = model_name
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.graph = tf.Graph()
        with self.graph.as_default():
            """Model"""
            self.seqInput = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.c_dim],name='predictor/input')
            self.score = self.PredictorNet(self.seqInput)
            self.label = tf.placeholder(tf.float32, shape=[None, 1],name='predictor/label')

            """Loss"""
            self.loss = tf.losses.mean_squared_error(self.label,self.score)
            self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
            self.saver = tf.train.Saver(max_to_keep=1)
            self.sess.run(tf.initialize_all_variables())
        return

    def Train(self,
              lr=1e-6,
              beta1=0.5,
              beta2=0.9,
              epoch=1000,
              earlystop=20,
              log_dir = 'log',
              checkpoint_dir='./predict_model'
              ):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        #
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with self.graph.as_default():
            self.epoch = epoch
            self.iteration = self.dataset_num // self.BATCH_SIZE
            self.earlystop = earlystop
            self.opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(self.loss)
            self.sess.run(tf.initialize_all_variables())
            
            counter = 1
            start_time = time.time()
            gen = self.inf_train_gen()
            best_R = -0.5
            convIter = 0
            for epoch in range(1, 1+self.epoch):
                # get batch data
                for idx in range(1, 1+self.iteration):
                    I = gen.__next__()
                    _, loss = self.sess.run([self.opt,self.loss],feed_dict={self.seqInput:self.x[I,:,:],self.label:self.y[I,:]})
                    # display training status
                    counter += 1
                    
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, loss))

                train_pred = self.Predictor(self.x,'oh')
                train_pred = np.reshape(train_pred,(train_pred.shape[0],1))
                train_R = pearsonr(train_pred,self.y)
                train_R_spearman,_=spearmanr(train_pred,self.y)
                val_pred = self.Predictor(self.val_x,'oh')
                val_pred = np.reshape(val_pred,(val_pred.shape[0],1))
                val_R = pearsonr(val_pred,self.val_y)
                val_R_spearman,_=spearmanr(val_pred,self.val_y)
                print('Epoch {}: train R: {}, val R: {}'.format(
                        epoch,
                        train_R,
                        val_R))
                
                print('Epoch {}: train R(spearman): {}, val R(spearman): {}'.format(
                        epoch,
                        train_R_spearman,
                        val_R_spearman))
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model

                # save model
                if val_R>best_R:
                    best_R = val_R
                    self.save(self.checkpoint_dir, counter)
                else:
                    convIter += 1
                    if convIter>=earlystop:
                        break
                      
        self.load()
        train_pred = self.Predictor(self.x,'oh')
        train_pred = np.reshape(train_pred,(train_pred.shape[0],1))
        train_R = pearsonr(train_pred,self.y)
        train_R_spearman,_=spearmanr(train_pred,self.y)
        val_pred = self.Predictor(self.val_x,'oh')
        val_pred = np.reshape(val_pred,(val_pred.shape[0],1))
        val_R = pearsonr(val_pred,self.val_y)
        val_R_spearman,_=spearmanr(val_pred,self.val_y)
        #Val:
        data_val = pd.DataFrame(columns=['Promoter strength by experiment','Predicted promoter strength'])
        data_val['Promoter strength by experiment']=np.squeeze(self.val_y)
        data_val['Predicted promoter strength']=np.squeeze(val_pred)
        pdf = PdfPages(log_dir+'/true_pred_scatter_val.pdf')
        plt.figure()
        #plt.scatter(self.val_y,val_pred)
        #plt.text(np.max(self.val_y)*0.1,np.max(val_pred)*0.9,'r='+str(val_R))
        sns.set(style="whitegrid",font_scale=1.2)
        g=sns.regplot(x='log2(Promoter strength by experiment)', y='log2(Predicted promoter strength)', data=data_val,
              color='#000000',
              marker='+',
              scatter_kws={'s': 40,'color':'g',},#设置散点属性，参考plt.scatter
              line_kws={'linestyle':'--','color':'r'})#设置线属性，参考 plt.plot 
        #y_hat_val=est.predict(self.val_y)
        #plt.scatter(self.val_y, y_hat_val, alpha=0.3) 
        plt.xlabel('log2(Promoter strength by experiment)')
        plt.ylabel('log2(Predicted promoter strength)')
        pdf.savefig()
        pdf.close()
        #Train:
        data_train = pd.DataFrame(columns=['log2(Promoter strength by experiment)','log2(Predicted promoter strength)'])
        data_train['log2(Promoter strength by experiment)']=np.squeeze(self.y)
        data_train['log2(Predicted promoter strength)']=np.squeeze(train_pred)
        pdf = PdfPages(log_dir+'/true_pred_scatter_train.pdf')
        plt.figure()
        #plt.scatter(self.y,train_pred) #
        #plt.text(np.max(self.y)*0.1,np.max(train_pred)*0.9,'r='+str(train_R))
        #y_hat_train=est.predict(self.y)
        #plt.scatter(self.y, y_hat_train, alpha=0.3) 
        g=sns.regplot(x='Promoter strength by experiment', y='Predicted promoter strength', data=data_train,
              color='#000000',
              marker='+',
              scatter_kws={'s': 40,'color':'g',},#设置散点属性，参考plt.scatter
              line_kws={'linestyle':'--','color':'r'})#设置线属性，参考 plt.plot 
        plt.xlabel('Promoter strength by experiment')
        plt.ylabel('Predicted promoter strength')
        pdf.savefig()
        pdf.close()
        return epoch

    def inf_train_gen(self):
        I = np.arange(self.dataset_num)
        while True:
            np.random.shuffle(I)
            for i in range(0, len(I)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield I[i:i+self.BATCH_SIZE]

    def save(self, checkpoint_dir, step):
        with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
            for c in self.charmap:
                f.write(c+'\t')
                
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)
        
    def load(self, checkpoint_dir = None, model_name = None):
        with self.graph.as_default():
            print(" [*] Reading checkpoints...")
            if checkpoint_dir == None:
                checkpoint_dir = self.checkpoint_dir
            if model_name == None:
                model_name = self.model_name
                
            with open(os.path.join(checkpoint_dir,model_name+'charmap.txt'),'r') as f:#checkpoint_dir+'/'+model_name +'charmap.txt','r') as f:
                self.invcharmap = str.split(f.read())
                self.charmap = {}
                i=0
                for c in self.invcharmap:
                    self.charmap[c] = i
                    i+=1
            
            checkpoint_dir = os.path.join(checkpoint_dir, model_name)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path) #cnn.model-16493
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
    
    def GradientDescent(self, oh):
        oh = oh.astype('float32')
        with self.graph.as_default():
            if hasattr(self,'oh_tensor') == False:
                self.oh_tensor = tf.Variable(oh,name='oh_tensor')
            self.sess.run(tf.assign(self.oh_tensor,oh))
            gradient = self.sess.run(tf.gradients(self.PredictorNet(self.oh_tensor,reuse=True),[self.oh_tensor])[0])
            return gradient

    def AIpromoter_scoring_filter_out(self,promoter_dir,input_filename,out_filename,Nout):
        seqs=self.file2seq('./',input_filename)
        filtered_out=open(os.path.join(promoter_dir, out_filename),'w')
        #for i in range(len(seqs)):
        y=self.Predictor(seqs)
        seq_exp=pd.DataFrame(columns=['promoter','predicted_strength'])
        seq_exp['promoter']=seqs[:len(y)]
        seq_exp['predicted_strength']=y
        seq_exp=seq_exp.sort_values(by=['predicted_strength'])
        #
        if Nout<=len(seq_exp):
            count=0
            print('writing out seqs...')
            for i in range(Nout):
                count+=1
                filtered_out.write('>'+str(count)+'\n')
                filtered_out.write(str(seq_exp['promoter'][i])+'\n')
        else:
            print('not enougth AI_designed promoters,please generate more seqs')
        filtered_out.close()
        print('[Done] Promoters written out.')
        return

    def file2seq(self,promoter_dir,filename):
        f=open(os.path.join(promoter_dir, filename))
        seqs=[]
        for line in f.readlines():
            line=line.strip('\n')
            if '>' not in line:
                seqs.append(line.strip('\n'))
        f.close()
        return seqs
        