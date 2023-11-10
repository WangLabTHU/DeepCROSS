import time
import tensorflow as tf
import keras
from keras import regularizers
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Reshape,Activation,concatenate
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout
#from ..ops import Conv1D, Linear, ResBlock
from ..ops.param import params_with_name
import math
import numpy as np
import os
from ..ProcessData import seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')

def pearsonr(x,y):
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    r = r_num / r_den
    return r
class CNN_crosssignal:
    def load_dataset(self,train_data,val_data=None):
        self.x,self.y = load_fun_data_exp3(train_data)
        self.charmap, self.invcharmap = GetCharMap(self.x)
        self.x = seq2oh(self.x,self.charmap)
        self.seq_len = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            np.random.seed(3)
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*7//10
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:] #训练集：7128
        self.dataset_num = self.x.shape[0]
        return 

    def PredictorNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Predictor", reuse=reuse):#regW = regularizers.l2(0.00001)
            regW = regularizers.l2(0.00001)
            x = Conv1D(self.DIM, 7, activation='relu',activity_regularizer = None)(x)
            x = MaxPooling1D(pool_size=3)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM*2, 5, activation='relu', activity_regularizer = None)(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv1D(self.DIM*4, 3, activation='relu', activity_regularizer = None)(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x=Flatten()(x)
            x = Dropout(0.2)(x)
            x = Dense(512,activation='relu', kernel_regularizer= regW)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x1 = Dense(1,activation='sigmoid',kernel_regularizer= regW)(x) #,kernel_regularizer= regW
            x2 = Dense(1,activation='sigmoid',kernel_regularizer= regW)(x)
            x3 = Dense(1,activation='sigmoid',kernel_regularizer= regW)(x)
            output = concatenate([x1,x2,x3],axis=-1) 
            return output
    
    def BuildModel(self,
                   DIM = 512,
                   kernel_size = 5,
                   batch_size=32,
                   model_name='cnn_cross'
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
            self.label = tf.placeholder(tf.float32, shape=[None, 3],name='predictor/label')

            """Loss"""
            self.loss = tf.keras.losses.binary_crossentropy(self.label,self.score)
            self.sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
            self.saver = tf.train.Saver(max_to_keep=1)
            self.sess.run(tf.initialize_all_variables())
        return

    def Train(self,
              lr=1e-4,
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

        with self.graph.as_default():
            self.epoch = epoch
            self.iteration = self.dataset_num // self.BATCH_SIZE
            self.earlystop = earlystop
            self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss)
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
                    # display training status
                    counter += 1
                    
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, np.mean(loss)))#---4.,loss

                train_pred = self.Predictor(self.x,'oh')
                train_R = pearsonr(train_pred,self.y)
                val_pred = self.Predictor(self.val_x,'oh')
                val_R = pearsonr(val_pred,self.val_y)
                print('Epoch {}: train R: {}, val R: {}'.format(
                        epoch,
                        train_R,
                        val_R))

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
            train_R = pearsonr(train_pred,self.y)
            val_pred = self.Predictor(self.val_x,'oh')
            val_R = pearsonr(val_pred,self.val_y)
            print(' [*] val_R:'+str(val_R))
            pdf = PdfPages(log_dir+'/true_pred_scatter.pdf')
            plt.figure()
            plt.scatter(val_pred,self.val_y,label='R = %0.2f' % val_R)
            plt.xlabel('True')
            plt.ylabel('Pred')
            pdf.savefig()
            pdf.close()
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
        with self.graph.as_default():
            print(" [*] Reading checkpoints...")
            if checkpoint_dir == None:
                checkpoint_dir = self.checkpoint_dir
            if model_name == None:
                model_name = self.model_name
                
            with open(checkpoint_dir+'/'+model_name +'charmap.txt','r') as f:
                self.invcharmap = str.split(f.read())
                self.charmap = {}
                i=0
                for c in self.invcharmap:
                    self.charmap[c] = i
                    i+=1
            
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
        return y
    
    def GradientDescent(self, oh):
        oh = oh.astype('float32')
        with self.graph.as_default():
            if hasattr(self,'oh_tensor') == False:
                self.oh_tensor = tf.Variable(oh,name='oh_tensor')
            self.sess.run(tf.assign(self.oh_tensor,oh))
            all=tf.gradients(self.PredictorNet(self.oh_tensor,reuse=True),[self.oh_tensor]) #(1,0)
            self.sess.run([tf.global_variables_initializer()])#, tf.global_variables_initializer()
            gradient = self.sess.run(all[0])#[0]
            return gradient     
####
#AUC
####
    def predict_val_value(self,checkpoint_dir='./weight_dir'):
        print(' [*] Predicting valid set...')
        y_pred = np.mean(self.Predictor(self.val_x,datatype='oh'),axis=1) #793,3 False
        y_true=self.val_y 
        y_true=np.array(y_true,dtype='int')
        y_trueuniv=[]
        y_preduniv=[]
        for i in range(len(y_true)):
            if np.sum(y_true[i,:])==3:
                y_trueuniv.append(1)
            else:
                y_trueuniv.append(0)
            y_preduniv.append(y_pred[i])
        pdf = PdfPages(checkpoint_dir+'crosssignal_auc_univ.pdf')  
        plt.figure(figsize=(10,10))
        fpr,tpr,threshold = roc_curve(y_trueuniv,y_preduniv) 
        roc_auc = auc(fpr,tpr) 
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        pdf.savefig() 
        pdf.close()
        print(' [*] Predicting valid set done(AUC)')
        #
        train_pred = self.Predictor(self.x,'oh')
        train_R = pearsonr(train_pred,self.y)
        val_pred = self.Predictor(self.val_x,'oh')
        val_R = pearsonr(val_pred,self.val_y)
        print(' [*] val_R:'+str(val_R))
        pdf = PdfPages(checkpoint_dir+'/true_pred_scatter.pdf')
        plt.figure()
        plt.scatter(val_pred,self.val_y,label='R = %0.2f' % val_R)
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.legend(loc="lower right")
        pdf.savefig()
        pdf.close()
        print(' [*] Evaluating predicting done(R value)')


        