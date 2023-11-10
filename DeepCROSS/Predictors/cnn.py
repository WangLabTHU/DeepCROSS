import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout

import math
import numpy as np
import os
from scipy.stats import pearsonr
from ..ProcessData import seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3

class CNN():
        
    def PredictorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):
            x = Conv1D(self.DIM, self.kernel_size, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM*2, self.kernel_size, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM*4, self.kernel_size, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            
            x=Flatten()(x)
            x = Dropout(0.2)(x)
            y = Dense(1)(x)
            return y
    
    def BuildModel(self,
                   train_data,
                   val_data=None,
                   DIM = 128,
                   kernel_size = 5,
                   batch_size=32,
                   checkpoint_dir='./predict_model',
                   model_name='cnn'
                   ):
        #self.x,self.y = load_fun_data(train_data)----1.
        self.x,self.y = load_fun_data_exp3(train_data,flag=1,already_log=True)
        self.y=np.reshape(self.y,(self.y.shape[0],1))
        self.charmap, self.invcharmap = GetCharMap(self.x)
        self.x = seq2oh(self.x,self.charmap)
        self.seq_len = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            #d = self.x.shape[0]//10 *9
            np.random.seed(3)
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*int(0.9*10)//10
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]#--2.[d:,:]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:]#
        self.dataset_num = self.x.shape[0]
        self.DIM = DIM
        self.kernel_size = kernel_size
        self.BATCH_SIZE = batch_size
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        """Model"""
        self.seqInput = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.c_dim],name='input')
        self.score = self.PredictorNet(self.seqInput)
        self.label = tf.placeholder(tf.float32, shape=[None,1],name='label')

        """Loss"""
        self.loss = tf.losses.mean_squared_error(self.label,self.score)
        
        self.saver = tf.train.Saver(max_to_keep=1)
        return

    def Train(self,
              lr=1e-4,
              beta1=0.5,
              beta2=0.9,
              epoch=1000,
              earlystop=20,
              ):
        
        self.epoch = epoch
        self.iteration = self.dataset_num // self.BATCH_SIZE
        self.earlystop = earlystop
        
        
        self.opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        
        counter = 1
        start_time = time.time()
        gen = self.inf_train_gen()
        best_R = 0
        convIter = 0
        for epoch in range(1, 1+self.epoch):
            # get batch data
            for idx in range(1, 1+self.iteration):
                I = gen.__next__()
                _, loss = self.sess.run([self.opt,self.loss],feed_dict={self.seqInput:self.x[I,:,:],self.label:self.y[I,:]})
                #---3.self.label:self.y[I,:]
                # display training status
                counter += 1
                
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, loss))

            train_pred = self.Predictor(self.x,'oh')
            train_pred = np.reshape(train_pred,(train_pred.shape[0],1))
            train_R = pearsonr(train_pred,self.y)[0]
            val_pred = self.Predictor(self.val_x,'oh')
            val_pred = np.reshape(val_pred,(val_pred.shape[0],1))
            val_R = pearsonr(val_pred,self.val_y)[0]
            print('Epoch {}: train R: {}, val R: {}'.format(
                    epoch,
                    train_R,
                    val_R))
            
            
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

        return

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
        print(" [*] Reading checkpoints...")
        if checkpoint_dir == None:
            checkpoint_dir = self.checkpoint_dir
        if model_name == None:
            model_name = self.model_name
            
        with open(checkpoint_dir+ '/' + model_name + 'charmap.txt','r') as f:
            self.invcharmap = str.split(f.read())
            self.charmap = {}
            i=0
            for c in self.invcharmap:
                self.charmap[c] = i
                i+=1
        
        #checkpoint_dir = os.path.join(checkpoint_dir, model_name)
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
    
def plot(real,pred,name):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(real,pred)
    plt.xlabel('True value')
    plt.ylabel('Predict value')
    plt.savefig(name+'.jpg')