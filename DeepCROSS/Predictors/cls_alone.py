import time
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
#from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization,Flatten, Dropout
from ..ops import Conv1D, Linear, ResBlock
from keras.layers import MaxPooling1D,UpSampling1D
from keras.layers import Bidirectional,LSTM,Dropout,BatchNormalization
#from keras.layers import *
from ..ops.param import params_with_name
import math
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from ..ProcessData import seq2oh,GetCharMap,load_fun_data,load_fun_data_exp3
from sklearn.metrics import precision_score,recall_score,f1_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dropout_rate = 0.2

class CLS_alone():
    def __init__(self):
        self.nb_blocks = 2
        self.filters = 24
        self.BATCH_SIZE =256
    #########
    #1.CNN+LSTM#
    #########
    def PredictorNet(self, x, is_training=True, reuse=False):
            output = Conv1D('Conv1D.1', self.c_dim, 128, 6, x)
            #BN1
            #output = BatchNormalization()(output)
            output = MaxPooling1D(pool_size=2)(output) 
            output = Conv1D('Conv1D.2', 128, 256, 3, output)
            #BN2
            #output = BatchNormalization()(output)
            output = MaxPooling1D(pool_size=2)(output)
            for i in range(1,1+8):#8个:4*8
                output = ResBlock(output, 256, 3, 'ResBlock.{}'.format(i))
            output = Conv1D('Conv_label.1', 256, 512, 3, output)
            #BN3
            #output = BatchNormalization()(output)
            output = Conv1D('Conv_label.2', 512, 512, 3, output)
            #BN4
            #output = BatchNormalization()(output)
            output = Conv1D('Conv_label.3', 512, 512, 3, output)
            #BN5
            #output = BatchNormalization()(output)
            output = tf.nn.relu(output)
            #output = tf.reshape(output, [-1, 512*int(self.SEQ_LEN/2/2)])#6.
            #output = Linear('Dense_label.1',512*int(self.SEQ_LEN/2/2),1024, output)#7.encoder_out
            #
            output = Dropout(0.2)(output) #保留的比例
            output = Bidirectional(LSTM(50,return_sequences=True))(output) #75 #(输入:bs,41,512,bs,41,100)
            output = Linear('Dense_label.1',41*50*2,512, output)
            output = Dropout(0.2)(output)
            #
            cls_logits_BS =  Linear('Dense_label.2',512,self.nbin, output)#7.
            cls_logits_EC =  Linear('Dense_label.3',512,self.nbin, output)#7.
            cls_logits_PA =  Linear('Dense_label.4',512,self.nbin, output)#7.
            return cls_logits_BS,cls_logits_EC,cls_logits_PA #cls_logits
    #########
    #2.U-net#
    #########
    # def PredictorNet(self, x, is_training=True, reuse=False):
    #     with tf.variable_scope("Predictor", reuse=reuse):
    #         #output= Conv1D('Conv1D.1', self.c_dim, 32, 7, x)#128通道
    #         output=tf.keras.layers.Conv1D(filters=32, kernel_size=7,strides=1, padding='same',activation='relu')(x)
    #         # output=tf.keras.layers.Conv1D(filters=32, kernel_size=3 strides=2, padding='same',activation='relu')(output)
    #         # output=tf.keras.layers.Conv1D(filters=32, kernel_size=3 strides=2, padding='same',activation='relu')(output)#41,32
    #         input_block = self.InputBlock(output, 32)#160,32(2*conv)
    #         # 下采样
    #         con_1 = self.ContractingPathBlock(input_block, 128)#80,128
    #         con_2 = self.ContractingPathBlock(con_1, 256)#40,256
    #         con_3 = self.ContractingPathBlock(con_2, 512)#20,512
    #         con_4 = self.ContractingPathBlock(con_3, 1024)#10,1024
    #         # 上采样
    #         exp_4 = self.ExpansivePathBlock(con_4, con_3, 512, 512)#10,1024->20,512#exp_4 #con_3
    #         exp_3 = self.ExpansivePathBlock(exp_4, con_2, 256, 256)#20,512->40,256
    #         exp_2 = self.ExpansivePathBlock(exp_3, con_1, 128, 128)#40,256->80,128
    #         exp_1 = self.ExpansivePathBlock(exp_2, input_block, 64, 64)#160,64
    #         output=tf.keras.layers.Flatten()(exp_1)
    #         cls_logits_BS =  Linear('Dense_label.2',160*64,self.nbin, output)#7.
    #         cls_logits_EC =  Linear('Dense_label.3',160*64,self.nbin, output)#7.
    #         cls_logits_PA =  Linear('Dense_label.4',160*64,self.nbin, output)#7.
    #         return cls_logits_BS,cls_logits_EC,cls_logits_PA #cls_logits

    # # UNet输入模块
    # def InputBlock(self,input, filters, kernel_size=3, strides=1, padding='same'):
    #     conv_1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=4, strides=strides, padding='valid',#kernel_size #padding
    #                                     activation='relu')(input)  # 卷积块1
    #     return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid',#padding
    #                                 activation='relu')(conv_1)  # 卷积块2


    # # 收缩路径模块
    # def ContractingPathBlock(self,input, filters, kernel_size=3, strides=1, padding='same'):
    #     down_sampling = tf.keras.layers.MaxPool1D(2)(input)  # 最大池化
    #     conv_1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #                                     activation='relu')(down_sampling)  # 卷积块1
    #     return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #                                 activation='relu')(conv_1)  # 卷积块2


    # # 扩张（恢复）路径模块
    # def ExpansivePathBlock(self,input, con_feature, filters, tran_filters, kernel_size=3, tran_kernel_size=2, strides=1,
    #                     tran_strides=2, padding='same', tran_padding='same'):
    #     upsampling = tf.keras.layers.UpSampling1D(2)(input) #20,1024, con_feature:20,512
    #     upsampling = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, strides=1, padding='same',activation='relu')(upsampling) 
    #     #tf.keras.layers.Conv1DTranspose(filters=tran_filters, kernel_size=tran_kernel_size,
    #                                                 #strides=tran_strides, padding=tran_padding)(input)  # 上采样（转置卷积方式）
    #     #####
    #     #padding_n = (con_feature.shape)[1] - (upsampling.shape)[1]
    #     #upsampling = tf.pad(upsampling, ((0), (0, padding_n), (0)), 'constant')
    #     #
    #     #padding_h = (con_feature.shape)[1] - (upsampling.shape)[1]
    #     #padding_w = (con_feature.shape)[2] - (upsampling.shape)[2]
    #     #upsampling = tf.pad(upsampling, ((0, 0), (0, padding_h), (0, padding_w), (0, 0)), 'constant')
    #     #con_feature = tf.image.resize(con_feature, upsampling.shape)[1],
    #                                 #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
    #     # con_feature = tf.image.resize(con_feature, ((upsampling.shape)[1], (upsampling.shape)[2]),
    #     #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 裁剪需要拼接的特征图
    #     concat_feature = tf.concat([con_feature, upsampling], axis=2)  #40,# 拼接扩张层和收缩层的特征图（skip connection）
    #     #####
    #     conv_1_expan = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #                                     activation='relu')(concat_feature)  # 卷积1
    #     return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #                                 activation='relu')(conv_1_expan)  # 卷积2
    #############
    #3.Dense-net#
    #############
    # def PredictorNet(self, x, is_training=True, reuse=False):
    #     with tf.variable_scope("Predictor", reuse=reuse):
    #         #output= Conv1D('Conv1D.1', self.c_dim, 32, 7, x)
    #         #output = tf.keras.layers.Conv1D(filters=2 * self.filters, kernel_size=7,strides=2, padding='same',activation='relu')(x)
    #         output = self.conv_layer(x, filter=2 * self.filters, kernel=7, stride=2, layer_name='conv0')#82,48
    #         output = self.Max_Pooling(output, pool_size=3, stride=1)#81,48

    #         output = self.dense_block(input_x=output, nb_layers=6, layer_name='dense_1',is_training=is_training)
    #         output = self.transition_layer(output, scope='trans_1',is_training=is_training)#1.40,96

    #         output = self.dense_block(input_x=output, nb_layers=12, layer_name='dense_2',is_training=is_training)
    #         output = self.transition_layer(output, scope='trans_2',is_training=is_training)#2.20,192

    #         output = self.dense_block(input_x=output, nb_layers=48, layer_name='dense_3',is_training=is_training)
    #         output = self.transition_layer(output, scope='trans_3',is_training=is_training)#3.10,672

    #         output = self.dense_block(input_x=output, nb_layers=32, layer_name='dense_final',is_training=is_training)#4.10,1440

    #         output = self.Batch_Normalization(output, training=is_training, scope='linear_batch')#self.training
    #         output = self.Relu(output)
    #         output = self.Global_Average_Pooling(output)#1,1440
    #         output = tf.layers.Flatten()(output)#1440
    #         #output = Linear(output)
    #         #
    #         cls_logits_BS =  Linear('Dense_label.2',1440,self.nbin, output)#
    #         cls_logits_EC =  Linear('Dense_label.3',1440,self.nbin, output)#
    #         cls_logits_PA =  Linear('Dense_label.4',1440,self.nbin, output)#
    #         return cls_logits_BS,cls_logits_EC,cls_logits_PA #cls_logits

    # def Global_Average_Pooling(self,x, stride=1):
    #     with tf.name_scope('Global_avg'):
    #         pool_size = x.get_shape()[1]#np.shape(x)[1]
    #         #width = np.shape(x)[1]
    #         #height = np.shape(x)[2]
    #         #pool_size = [width, height]
    #         return tf.layers.average_pooling1d(inputs=x, pool_size=[pool_size], strides=stride) 

    # def transition_layer(self, x, scope,is_training):
    #     with tf.name_scope(scope):
    #         x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch1')#self.training
    #         x = self.Relu(x)
    #         shape = x.get_shape().as_list()
    #         in_channel = shape[2]#3
    #         #x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
    #         x = self.conv_layer(x, filter=in_channel*0.5, kernel=1, layer_name=scope+'_conv1')#[1,1]
    #         x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
    #         x = self.Average_pooling(x, pool_size=2, stride=2)#pool_size[2,2]
    #         return x

    # def bottleneck_layer(self, x, scope,is_training):
    #     with tf.name_scope(scope):
    #         x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch1')#self.training
    #         x = self.Relu(x)
    #         x = self.conv_layer(x, filter=4 * self.filters, kernel=1, layer_name=scope+'_conv1')#[1,1]
    #         x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
    #         x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch2')#self.training
    #         x = self.Relu(x)
    #         x = self.conv_layer(x, filter=self.filters, kernel=3, layer_name=scope+'_conv2')#[3,3]
    #         x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
    #         return x

    # def dense_block(self, input_x, nb_layers, layer_name,is_training):
    #     with tf.name_scope(layer_name):
    #         layers_concat = list()
    #         layers_concat.append(input_x)
    #         x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0),is_training=is_training)
    #         layers_concat.append(x)
    #         for i in range(nb_layers - 1):
    #             x = self.Concatenation(layers_concat)
    #             x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1),is_training=is_training)
    #             layers_concat.append(x)
    #         x = self.Concatenation(layers_concat)
    #         return x

    # def Batch_Normalization(self,x, training, scope):
    #     with arg_scope([batch_norm],
    #                 scope=scope,
    #                 updates_collections=None,
    #                 decay=0.9,
    #                 center=True,
    #                 scale=True,
    #                 zero_debias_moving_mean=True) :
    #         return tf.cond(training,
    #                     lambda : batch_norm(inputs=x, is_training=training,reuse=None),#is_training=training,, reuse=None,#tf.constant(True, dtype=tf.bool)#training,
    #                     lambda : batch_norm(inputs=x, is_training=training,reuse=True))#is_training=training,

    # def Batch_Normalization_2(self,x, training, scope):
    #     with tf.name_scope(scope):
    #         return tf.layers.batch_normalization(inputs=x,training=training)


    # def conv_layer(self,input, filter, kernel, stride=1, layer_name="conv"):
    #     with tf.name_scope(layer_name):
    #         network = tf.layers.conv1d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
    #         return network

    # def Drop_out(self,x, rate, training):
    #     return tf.layers.dropout(inputs=x, rate=rate, training=training)

    # def Concatenation(self,layers):
    #     return tf.concat(layers, axis=2)#axis=3

    # def Relu(self,x):
    #     return tf.nn.relu(x)

    # def Max_Pooling(self,x, pool_size=3, stride=2, padding='VALID'):
    #     return tf.layers.max_pooling1d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    # def Average_pooling(self,x, pool_size=2, stride=2,padding='VALID'):
    #     return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=stride,padding=padding) 
    #AveragePooling1D

    def get_cls_loss(self,BS_logits,EC_logits,PA_logits): 
        BS=tf.expand_dims(BS_logits, 1)
        EC=tf.expand_dims(EC_logits, 1)
        PA=tf.expand_dims(PA_logits, 1)    
        logits = tf.concat((BS,EC,PA),axis=1)
        labels=tf.cast(self.label,tf.int64)#多分类:(bs,3)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(##tf.nn.sigmoid_cross_entropy_with_logits(#
            labels=labels,
            logits=logits,
            name='cross_entropy')
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def get_cls_accuracy(self):
        labels = self.label #(128,3)#cls_predict = tf.concat((self.layers['y_BS'].expand_dims(one_img, -1),self.layers['y_EC'].expand_dims(one_img, -1),self.layers['y_PA'].expand_dims(one_img, -1)),axis=-1)#(128,) #384
        BSresult = tf.expand_dims(self.BS_y,1)#self.layers['y_BS']
        ECresult = tf.expand_dims(self.EC_y,1)#self.layers['y_EC']
        PAresult = tf.expand_dims(self.PA_y,1)#self.layers['y_PA']
        cls_predict = tf.concat((BSresult,ECresult,PAresult),axis=-1)#128,3
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
        onehot = []
        label = []
        exp_all=[]
        for i in range(len(seq_exp)):
            exp_all.append(seq_exp[i][1])
        exp_all=sum(exp_all,[])
        #把exp_all分为n个bin，画个图
        pdf = PdfPages(os.path.join(log_dir, 'expdistri_forbin.pdf')) 
        plt.figure(1,figsize=(7,7)) 
        n, binlist, patches=plt.hist(exp_all, bins=self.nbin, color='green',alpha=0.5, rwidth=0.85,label='PA_exp')
        pdf.savefig() 
        pdf.close()
        if self.nbin==5:
            binlist=[-20,-10,-5,0,5,10] #去掉了-15这个bin
        else:
            binlist=list(binlist)#3个bin就按照正常分即可
        print('Expression bins:')
        print(binlist)
        binlist[-1]=binlist[-1]+1
        exp_bin=[]
        for i in range(len(seq_exp)):
            BS_1=seq_exp[i][1][0]#np.log2(seq_exp['tx_norm_BS'][i]+10**(-6))
            EC_1=seq_exp[i][1][1]#np.log2(seq_exp['tx_norm_EC'][i]+10**(-6))
            PA_1=seq_exp[i][1][2]#np.log2(seq_exp['tx_norm_PA'][i]+10**(-6))
            index_BS=np.where(BS_1>=binlist)[0][-1]
            index_EC=np.where(EC_1>=binlist)[0][-1]
            index_PA=np.where(PA_1>=binlist)[0][-1]#[-1]
            exp_bin.append([index_BS,index_EC,index_PA])
        for i in range(len(seq_exp)):
            seq=seq_exp[i][0]
            exp=exp_bin[i]
            #exp=seq_exp[i][1]
            eachseq = np.zeros([len(seq),4],dtype = 'float')
            for j in range(len(seq)):
                base=seq[j]
                eachseq[j,self.charmap[base]] = 1
            onehot.append(eachseq)
            label.append(np.array(exp))
        self.x = np.array(onehot) #self.seq_list
        self.y = np.squeeze(label) #self.label_list (10185,3)
        print(np.unique(self.y,axis=0))
        print(len(np.unique(self.y,axis=0))) #27
        print(self.y.shape)
        #self.x,self.y = load_fun_data_exp3(train_data,flag=1,already_log=True)
        #self.charmap, self.invcharmap = GetCharMap(self.x)
        #self.x = seq2oh(self.x,self.charmap)
        self.y=np.reshape(self.y,(self.y.shape[0],3))
        self.SEQ_LEN = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            #d = self.x.shape[0]//10 *9
            #5.10去掉的
            #np.random.seed(3)
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*int(0.9*10)//10
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]#--2.[d:,:]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:]#
        self.dataset_num = self.x.shape[0]
        self.dataset_num_val =self.val_x.shape[0]
        return
        
    def BuildModel(self,
                   DIM = 256,
                   kernel_size = 3
                   ):
        self.DIM = DIM
        self.kernel_size = kernel_size
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        """Model"""
        self.training = tf.placeholder(tf.bool)#
        self.seqInput = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
        self.score_BS,self.score_EC,self.score_PA = self.PredictorNet(self.seqInput,self.training)#.model
        #1.
        self.label = tf.placeholder(tf.float32, shape=[None,3],name='label')
        self.BS_y = tf.argmax(self.score_BS, axis=-1,name='label_predict_BS') 
        self.EC_y = tf.argmax(self.score_EC, axis=-1,name='label_predict_EC') 
        self.PA_y = tf.argmax(self.score_PA, axis=-1,name='label_predict_PA') 
        """Loss"""
        #2.
        #self.loss = tf.losses.mean_squared_error(self.label,self.score)
        self.loss = self.get_cls_loss(self.score_BS,self.score_EC,self.score_PA )
        self.saver = tf.train.Saver(max_to_keep=1)
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
            if cls_accuracy_val>best_percent:
                best_percent = cls_accuracy_val
                self.save(self.checkpoint_dir, counter)
            else:
                convIter += 1
                if convIter>=earlystop:
                    break
            #save
            self.save(self.checkpoint_dir, counter)
        return

    def inf_train_gen(self,num):
        I = np.arange(num)
        while True:
            #np.random.seed(3)
            np.random.shuffle(I)
            for i in range(0, len(I)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield I[i:i+self.BATCH_SIZE]

    def save(self, checkpoint_dir, step):
        # with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
        #     for c in self.charmap:
        #         f.write(c+'\t')
                
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
            
        # with open(checkpoint_dir+ '/' + model_name + 'charmap.txt','r') as f:
        #     self.invcharmap = str.split(f.read())
        #     self.charmap = {}
        #     i=0
        #     for c in self.invcharmap:
        #         self.charmap[c] = i
        #         i+=1
        
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
        I_val = gen_val.__next__()
        BS_pred,EC_pred,PA_pred = self.sess.run([self.BS_y,self.EC_y,self.PA_y],feed_dict={self.seqInput:self.val_x[I_val,:,:]})
        label_true=self.val_y[I_val,:]
        BS_true,EC_true,PA_true = np.hsplit(label_true,3) #(256,1)
        return  BS_pred,EC_pred,PA_pred,BS_true,EC_true,PA_true #(256,1): 数字,不是onehot标签

    def Evalu_acc(self,repeatN,out_dir):
        print('[*] Cal precision,recall and f1 ...')
        gen_val = self.inf_train_gen(self.dataset_num_val)
        BS_pred,EC_pred,PA_pred,BS_true,EC_true,PA_true=self.get_pred_ture(gen_val)
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
        precision_macro_BS=precision_score(BS_true, BS_pred,average='macro')
        recall_macro_BS=recall_score(BS_true, BS_pred,average='macro')
        f1score_macro_BS=f1_score(BS_true, BS_pred,average='macro')
                
        precision_macro_EC=precision_score(EC_true, EC_pred,average='macro')
        recall_macro_EC=recall_score(EC_true, EC_pred,average='macro')
        f1score_macro_EC=f1_score(EC_true, EC_pred,average='macro')

        precision_macro_PA=precision_score(PA_true, PA_pred,average='macro')
        recall_macro_PA=recall_score(PA_true, PA_pred,average='macro')
        f1score_macro_PA=f1_score(PA_true, PA_pred,average='macro')

	    #micro
        precision_micro_BS=precision_score(BS_true, BS_pred,average='micro')
        recall_micro_BS=recall_score(BS_true, BS_pred,average='micro')
        f1score_micro_BS=f1_score(BS_true, BS_pred,average='micro')

        precision_micro_EC=precision_score(EC_true, EC_pred,average='micro')
        recall_micro_EC=recall_score(EC_true, EC_pred,average='micro')
        f1score_micro_EC=f1_score(EC_true, EC_pred,average='micro')

        precision_micro_PA=precision_score(PA_true, PA_pred,average='micro')
        recall_micro_PA=recall_score(PA_true, PA_pred,average='micro')
        f1score_micro_PA=f1_score(PA_true, PA_pred,average='micro')
        #
        acc_out_macro=pd.DataFrame([[precision_macro_BS,recall_macro_BS,f1score_macro_BS],[precision_macro_EC,recall_macro_EC,f1score_macro_EC],[precision_macro_PA,recall_macro_PA,f1score_macro_PA]],columns=['pr','rc','f1'],index=['BS','EC','PA'])
        acc_out_macro.to_csv(out_dir+'/pr_rc_f1_macro_n'+str(repeatN)+'.csv')#,header=0,index=0
        #
        acc_out_micro=pd.DataFrame([[precision_micro_BS,recall_micro_BS,f1score_micro_BS],[precision_micro_EC,recall_micro_EC,f1score_micro_EC],[precision_micro_PA,recall_micro_PA,f1score_micro_PA]],columns=['pr','rc','f1'],index=['BS','EC','PA'])
        acc_out_micro.to_csv(out_dir+'/pr_rc_f1_micro_n'+str(repeatN)+'.csv')#,header=0,index=0
        return 0
