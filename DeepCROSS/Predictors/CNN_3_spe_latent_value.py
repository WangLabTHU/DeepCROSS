import sys
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Reshape,Activation,concatenate
from keras.layers.core import Flatten, Dropout,Lambda
from keras.models import Model
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from keras import regularizers
from keras import backend as K
import os
from ..ProcessData import seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3
#
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def prvalue(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(xm*xm) * K.sum(ym*ym))
    r = r_num / r_den
    return r

def prvalue_bin(y_true, y_pred):#(n,3,7)
    I_max_true=K.argmax(y_true,axis=2)#(n,3) ,K.argmax没有keepdims=True这个参数,K.max又不能返回index
    I_max_pred=K.argmax(y_pred,axis=2)#(n,3)
    sameN=np.sum(np.array(I_max_true==I_max_pred,dtype='float32'),axis=1) #(n,)
    spe3_all_rightN=np.sum(np.array(sameN>0,dtype='float32'))
    prop_all_right=spe3_all_rightN/tf.shape(y_true)[0]
    return prop_all_right



class CNN_exp_latent_value():
    def __init__(self,
                 train_data,
                 val_data=None,
                 DIM = 512,#128,
                 kernel_size = 5
                 ):
        self.x,self.y = load_fun_data_exp3(train_data)
        self.charmap, self.invcharmap = GetCharMap(self.x)
        #for bin and value
        #self.x = seq2oh(self.x,self.charmap)
        #for latent space input
        self.x =np.array(self.x).reshape(-1,64,1)
        #self.seq_len = self.x.shape[1]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            ###shuffle added by wy#begin
            np.random.seed(3)
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*9//10
            d = seq_index_A[:n]
            #####end
            ######origin begin
            #d = self.x.shape[0]//10 *9
            #self.val_x, self.val_y = self.x[d:,:,:], self.y[d:]
            #self.x, self.y = self.x[:d,:,:], self.y[:d]
            #########end
            ###for exp as single value by wy:
            #self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:]]
            #self.x, self.y = self.x[d,:,:], self.y[d]
            ###for exp as bins by wy
            #self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:,:]
            #self.x, self.y = self.x[d,:,:], self.y[d,:,:]
            ###for latent space as input
            self.val_x, self.val_y = self.x[seq_index_A[n:],:], self.y[seq_index_A[n:]]
            self.x, self.y = self.x[d,:], self.y[d]
        self.DIM = DIM
        self.kernel_size = 5
        self.model = self.BuildModel()

    def BuildModel_orig(self):
        regW = regularizers.l2(0.00001)
        #seqInput = Input(shape=(self.seq_len,4),name='seqInput')
        seqInput = Input(shape=(64,1),name='seqInput')
        #7,5,3
        x = Conv1D(self.DIM, 7, activation='relu', activity_regularizer = None)(seqInput)
        x = MaxPooling1D(pool_size=3)(x)
        x = BatchNormalization()(x)
        #seqInput!!!!!!!!!!!!!
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
        #used to be this:
        #y = Dense(1,kernel_regularizer= regW)(x) 
        #following is changed by wy
        #---1.order:BS,EC,PA
        x1 = Dense(1,kernel_regularizer= regW)(x) #(batch size,1) 
        x2 = Dense(1,kernel_regularizer= regW)(x)
        x3 = Dense(1,kernel_regularizer= regW)(x)
        #---2.
        # x1 = Dense(128,activation='relu',kernel_regularizer= regW)(x)
        # x1 = BatchNormalization()(x1)
        # x1 = Dense(1,kernel_regularizer= regW)(x1)#(batch size,1) 
        
        # x2 = Dense(128,activation='relu',kernel_regularizer= regW)(x)
        # x2 = BatchNormalization()(x2)
        # x2 = Dense(1,kernel_regularizer= regW)(x2)
        
        # x3 = Dense(128,activation='relu',kernel_regularizer= regW)(x)
        # x3 = BatchNormalization()(x3)
        # x3 = Dense(1,kernel_regularizer= regW)(x3)
        #both1&2
        y = concatenate([x1,x2,x3],axis=-1) 
        # up is changed by wy
        return Model(inputs=[seqInput],outputs=[y]) #(batch size,3) 

    def BuildModel(self):
        regW = regularizers.l2(0.00001)
        #seqInput = Input(shape=(self.seq_len,4),name='seqInput')
        seqInput = Input(shape=(64,1),name='seqInput')
        #7,5,3
        x = Conv1D(self.DIM, 7, activation='relu', activity_regularizer = None)(seqInput)
        x = MaxPooling1D(pool_size=3)(x)
        x = BatchNormalization()(x)
        #
        x1 = Conv1D(self.DIM*2, 5, activation='relu', activity_regularizer = None)(x)
        x1 = MaxPooling1D(pool_size=2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(self.DIM*4, 3, activation='relu', activity_regularizer = None)(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)
        x1 = BatchNormalization()(x1)
        x1=Flatten()(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Dense(512,activation='relu', kernel_regularizer= regW)(x1)
        x1 = BatchNormalization()(x1)
        #
        x2 = Conv1D(self.DIM*2, 5, activation='relu', activity_regularizer = None)(x)
        x2 = MaxPooling1D(pool_size=2)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(self.DIM*4, 3, activation='relu', activity_regularizer = None)(x2)
        x2 = MaxPooling1D(pool_size=2)(x2)
        x2 = BatchNormalization()(x2)
        x2=Flatten()(x2)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(512,activation='relu', kernel_regularizer= regW)(x2)
        x2 = BatchNormalization()(x2)
        #
        x3 = Conv1D(self.DIM*2, 5, activation='relu', activity_regularizer = None)(x)
        x3 = MaxPooling1D(pool_size=2)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(self.DIM*4, 3, activation='relu', activity_regularizer = None)(x3)
        x3 = MaxPooling1D(pool_size=2)(x3)
        x3 = BatchNormalization()(x3)
        x3=Flatten()(x3)
        x3 = Dropout(0.2)(x3)
        x3 = Dense(512,activation='relu', kernel_regularizer= regW)(x3)
        x3 = BatchNormalization()(x3)
        #
        x1 = Dense(1,kernel_regularizer= regW)(x1)#(batch size,1) 
        x2 = Dense(1,kernel_regularizer= regW)(x2)
        x3 = Dense(1,kernel_regularizer= regW)(x3)
        #
        y = concatenate([x1,x2,x3],axis=-1) 
        # up is changed by wy
        return Model(inputs=[seqInput],outputs=[y]) #(batch size,3) 

    def Train(self,
              lr=0.005,#1e-4,
              beta1=0.5,
              beta2=0.9,
              batch_size=32,
              epoch=1000,
              earlystop=20,
              weight_dir='./predict_model',
              model_name='predictor',
              ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.earlystop = earlystop
        self.weight_dir = weight_dir
        if os.path.exists(self.weight_dir) == False:
            os.makedirs(self.weight_dir)
        self.model_name = model_name
        
        self.opt = optimizers.Adam(lr=lr)#,beta_1=beta1,beta_2=beta2
        self.model.compile(optimizer=self.opt,loss='mse',metrics = [prvalue]) ## #'categorical_crossentropy' prvalue_bin
        json_string = self.model.to_json()
        open(weight_dir+model_name+'.json','w').write(json_string)
        self.model.fit([self.x],[self.y],
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       verbose=1,
                       validation_data=(self.val_x,self.val_y),
                       callbacks=[EarlyStopping(patience=self.earlystop),
                                  ModelCheckpoint(filepath=self.weight_dir+'/weight.h5',save_best_only=True)])
#        self.model.load_weights(self.weight_dir+'/weight.h5')
#        pred = self.model.predict(self.val_x)
#        plot(self.val_y,pred,'expression')
        return

    def load(self,weight_dir=None):
        if weight_dir is None:
            weight_dir = self.weight_dir
        self.model.load_weights(weight_dir+'/weight.h5')
        return
    
    def Predictor(self,seq):
        oh = seq2oh(seq,self.charmap)
        y = self.model.predict(oh)
        y = np.reshape(y,[y.shape[0]])
        return y

    def predict_val_value(self,train_data_now=None):
        print('Predicting valid set...')
        #validation set:1132条 
        y_pred=self.model.predict([self.val_x])#,batch_size=self.batch_size
        y_true=self.val_y #(1132,3)
        #3.Draw hist
        spe_list=['BS','EC','PA']
        color_list=['red','orange','blue']
        #'./exp_3_dir_EC_value/'
        pdf = PdfPages(self.weight_dir+'/exp_latent_value_true_pred_value'+'.pdf') 
        plt.figure(figsize=(21,7)) 
        for i in range(len(spe_list)):
            spe=spe_list[i]
            subplotI=int('13'+str(i+1))
            plt.subplot(subplotI)#131 132 133
            #
            bigdotnum = int(len(y_pred)*0.3)
            Bigdot_index = list(np.random.randint(0,len(y_pred)-1,bigdotnum)) #500
            plt.scatter(x=y_true[Bigdot_index,i],y=y_pred[Bigdot_index,i],c=color_list[i],s=3,label=spe)
            plt.scatter(x=y_true[:,i],y=y_pred[:,i],c=color_list[i],alpha=0.2,s=1) # np.random.uniform(0,1,len(val_label))
            plt.ylabel('Predicted activity ($\mathregular{log_2}$)')
            plt.xlabel('Observed activity ($\mathregular{log_2}$)')
            parameter = np.polyfit(y_true[:,i],y_true[:,i], 1) # n=1为一次函数，返回函数参数
            f = np.poly1d(parameter) # 拼接方程
            plt.plot(np.arange(-20,10,.2), f(np.arange(-20,10,.2)),'k--',linewidth=1.0,alpha=0.4)
            #
            #plt.scatter(y_true[:,i],y_pred[:,i],color=color_list[i],s=3,label=spe)
            r=pearsonr(y_true[:,i],y_pred[:,i])[0]
            print('r of '+spe+':')
            print(r)
            r=("%.3f" % r )
            plt.text(max(y_true[:,i])*0.6,max(y_pred[:,i])*0.8,r'r='+str(r))
            plt.legend(loc='upper right', fontsize=10, frameon=False, fancybox=False, borderpad=0.3,ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=2)
        pdf.savefig() 
        pdf.close()

def plot(real,pred,name):
    #import matplotlib.pyplot as plt
    #plt.clf()
    plt.scatter(real,pred)
    plt.xlabel('True value')
    plt.ylabel('Predict value')
    plt.savefig(name+'.jpg')