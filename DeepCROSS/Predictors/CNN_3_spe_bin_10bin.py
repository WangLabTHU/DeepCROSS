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



class CNN_exp_bin():
    def __init__(self,
                 train_data,
                 val_data=None,
                 DIM = 128,
                 kernel_size = 5
                 ):
        ###---1：看看怎么存seq和y,以及load y (11318,2)
        self.x,self.y = load_fun_data_exp3(train_data)
        self.charmap, self.invcharmap = GetCharMap(self.x)
        self.x = seq2oh(self.x,self.charmap)
        self.seq_len = self.x.shape[1]
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
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:,:]
            self.x, self.y = self.x[d,:,:], self.y[d,:,:]
        self.DIM = DIM
        self.kernel_size = 5
        self.model = self.BuildModel()

    def BuildModel(self):
        regW = regularizers.l2(0.00001)
        seqInput = Input(shape=(self.seq_len,4),name='seqInput')
        
        x = Conv1D(self.DIM, 7, activation='relu', activity_regularizer = None)(seqInput)
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
        #used to be this:
        #y = Dense(1,kernel_regularizer= regW)(x) 
        #following is changed by wy
        #order:BS,EC,PA
        x1 = Dense(5,activation='softmax',kernel_regularizer= regW)(x) #(batch size,6) 
        x2 = Dense(5,activation='softmax',kernel_regularizer= regW)(x)
        x3 = Dense(5,activation='softmax',kernel_regularizer= regW)(x)
        #
        x1 = Reshape((1,5))(x1)
        x2 = Reshape((1,5))(x2)
        x3 = Reshape((1,5))(x3)
        #
        y = concatenate([x1,x2,x3],axis=1) #axis=2
        # up is changed by wy
        return Model(inputs=[seqInput],outputs=[y]) 

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
        self.model.compile(optimizer=self.opt,loss='categorical_crossentropy',metrics = [prvalue]) ##'mse' prvalue_bin
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

    def predict_val(self,data=None):
        print('Predicting valid set...')
        #for cond7:
        x,y = load_fun_data_exp3(data)
        charmap2, invcharmap2= GetCharMap(x)
        x= seq2oh(x,charmap2)
        y_pred=self.model.predict([x])#,batch_size=self.batch_size
        y_true=y
        #1.Draw hist
        spe_list=['BS','EC','PA']
        color_list=['red','orange','blue']
        signal_list=['onlyPA','univ','EC_PA','onlyEC','BS_PA','BS_EC','onlyBS']
        #binlist=[-20,-15,-10,-5,0,5,10]
        binlist=[-20,-10,-5,0,5,10]
        if len(np.where(y_true[0][0]==1)[0])!=1: print('Error:More than one place with 1')
        pdf = PdfPages('./exp_3_dir_bin_only5/'+'exp_bin_true_pred_bin_cond7'+'.pdf') 
        plt.figure(figsize=(21,49)) 
        for j in range(7): #j表示序列
            for i in range(len(spe_list)): #i表示物种
                spe=spe_list[i]
                #subplotI=int('73'+str(3*j+i+1))
                #plt.subplot(subplotI)#131 132 133
                plt.subplot2grid((7,3),(j,i))
                plt.bar(np.array(binlist[:-1]),y_pred[j][i], align='edge',facecolor = color_list[i], width = 1.2,edgecolor = 'white',label='Pred_exp_'+spe)
                plt.bar(np.array(binlist[:-1]),y_true[j][i], align='edge',facecolor = 'green',alpha=0.3,width = 1.2,edgecolor = 'white',label='True_exp_'+spe)
                plt.legend(loc='upper right', fontsize=10, frameon=False, fancybox=False, borderpad=0.3,ncol=1, markerfirst=True, markerscale=4, numpoints=1, handlelength=2)
                plt.text(np.max(np.array(binlist[:-1]))*0.6,np.max(y_pred[j][i])*0.8,str(signal_list[j]))
                plt.xlabel('Promoter activity ($\mathregular{log_2}$)')
                plt.ylabel('Probability')
        pdf.savefig() 
        pdf.close()

        #For validation set 
        # x=self.val_x
        # y=self.val_y
        # y_pred=self.model.predict([x])#,batch_size=self.batch_size
        # y_true=y
        # I_max_true=np.argmax(y_true,axis=2)#(n,3) 
        # I_max_pred=np.argmax(y_pred,axis=2)#(n,3)
        # #1.active or no-active predict
        # active_correct0=0
        # active_correct1=0
        # active_correct2=0
        # active_correct3=0
        # for i in range(len(I_max_true)): #每条序列，在每个物种中是落到第几个bin
        #     activeI_true=np.where(I_max_true[i]!=0)[0]
        #     activeI_pred=np.where(I_max_pred[i]!=0)[0]
        #     allright_limit=max(len(activeI_true),len(activeI_pred))
        #     if np.sum(np.array(activeI_true==activeI_pred,dtype='float32'))==allright_limit: active_correct3+=1
        #     #if np.sum(np.array(activeI_true==activeI_pred,dtype='float32'))==allright_limit: active_correct2+=1
        #     #if np.sum(np.array(activeI_true==activeI_pred,dtype='float32'))==1: active_correct1+=1
        #     #if np.sum(np.array(activeI_true==activeI_pred,dtype='float32'))==0: active_correct0+=1
        # #N_active=np.array([active_correct0,active_correct1,active_correct2,active_correct3])
        # #prop_active=N_active/len(y_pred)
        # print('Active correct prop:')
        # #print(prop_active)
        # print(active_correct3/len(y_pred))

        # #2.highest right bin predict
        # same_matrix=np.array(I_max_true==I_max_pred,dtype='float32')#(n,3)
        # #具体BS,EC,PA对了多少,same_matrix==1就表示相等，即预测正确
        # print('predicted right loc shape:(2,x)')
        # print(np.array(np.where(same_matrix==1),dtype='float32').shape)
        # speloc=np.array(np.where(same_matrix==1),dtype='float32')[1]
        # #BS:0
        # BS_correct_num=np.sum(np.array(speloc==0,dtype='float32'))
        # #EC:1
        # EC_correct_num=np.sum(np.array(speloc==1,dtype='float32'))
        # #PA:2
        # PA_correct_num=np.sum(np.array(speloc==2,dtype='float32'))
        # N_3_spe=np.array([BS_correct_num,EC_correct_num,PA_correct_num])
        # prop_3_spe = N_3_spe/len(y_pred)
        # print('valid right bin prop(BS,EC,PA):')
        # print(prop_3_spe)

        # #总体计数
        # sameN=np.sum(same_matrix,axis=1) #(n,)
        # spe3_all_rightN=np.sum(np.array(sameN==3,dtype='float32'))
        # spe2_rightN=np.sum(np.array(sameN==2,dtype='float32'))
        # spe1_rightN=np.sum(np.array(sameN==1,dtype='float32'))
        # spe0_rightN=np.sum(np.array(sameN==0,dtype='float32'))
        # N=np.array([spe0_rightN,spe1_rightN,spe2_rightN,spe3_all_rightN])
        # prop=N/len(y_pred)
        # print('valid right bin prop(0,1,2,3):')
        # print(prop)


    




def plot(real,pred,name):
    #import matplotlib.pyplot as plt
    #plt.clf()
    plt.scatter(real,pred)
    plt.xlabel('True value')
    plt.ylabel('Predict value')
    plt.savefig(name+'.jpg')