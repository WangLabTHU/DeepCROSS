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
from tensorflow.keras.layers import Bidirectional,Conv1D,Activation,MaxPooling1D,Dense,Dropout,BatchNormalization,Flatten,AveragePooling1D,LSTM,Reshape,Dense,Flatten
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
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


class CLS_alone_optpredictor_denselstm():
    def __init__(self,Nseed=1,n=1,attention_num_blocks=5):
        self.nb_blocks = 2
        self.filters = 24
        self.BATCH_SIZE =256
        self.d_model = 64
        self.c_dim = 4
        self.attention_num_blocks = attention_num_blocks#3
        self.Nseed = Nseed
        self.n = n

    def PredictorNet(self, x,is_training=True, reuse=False,block_config=(2, 2, 4, 2)):
        #features0
        #enc = Conv1D('Conv1D.'+str(self.n), self.c_dim, filters = self.d_model, kernel_size=7, strides=1,padding=3,use_bias=False,x) 
        enc = tf.layers.conv1d(inputs=x, use_bias=False, filters=self.d_model, kernel_size=7, strides=1, padding='SAME')#padding=3#（bs,165,4）-> (bs,165,512)
        enc = BatchNormalization()(enc)
        enc = tf.nn.relu(enc)
        enc = MaxPooling1D(pool_size=3,strides=2,padding='SAME')(enc) #padding=1
        #enc = K.permute_dimensions(enc,(0, 2, 1))
        enc=Bidirectional(LSTM(self.d_model,return_sequences=True))(enc)
        enc=Bidirectional(LSTM(self.d_model,return_sequences=True))(enc)
        output=Bidirectional(LSTM(self.d_model,return_sequences=True))(enc)
        #enc = K.permute_dimensions(enc,(0, 2, 1))
        length = np.floor((165 + 2 * 1 - 1 - 2)/2 + 1)
        for i, num_layers in enumerate(block_config):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                output = self.dense_block(input_x=output, nb_layers=6, layer_name='dense_{}'.format(i),is_training=is_training)
            self.d_model = self.d_model + num_layers * 32 #growth_rate
            if i != len(block_config) - 1:
                output = self.transition_layer(output, scope='trans_{}'.format(i),is_training=True)
                self.d_model = self.d_model // 2
                length = np.floor((length - 1 - 1) / 2 + 1)
        output = self.Average_pooling(output, pool_size=2, stride=2,padding='SAME')#(bs,5,376) 
        output = Flatten()(output)#Reshape((5*376,-1)) output.get_shape().as_list()[1]
        out =  Dense(1) (output)
        return out

    #组件
    def Batch_Normalization1(self,x, training, scope):
        with arg_scope([batch_norm],
                    scope=scope,
                    updates_collections=None,
                    decay=0.9,
                    center=True,
                    scale=True,
                    zero_debias_moving_mean=True) :
            #training = tf.constant(training,dtype = tf.bool)
            return tf.cond(training,
                        lambda : batch_norm(inputs=x, is_training=training,reuse=None),#is_training=training,, reuse=None,#tf.constant(True, dtype=tf.bool)#training,
                        lambda : batch_norm(inputs=x, is_training=training,reuse=True))#is_training=training,

    def Batch_Normalization(self,x, training, scope):
         with tf.name_scope(scope):
             return tf.layers.batch_normalization(inputs=x,training=training)

    def conv_layer(self,input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv1d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
            return network

    def Average_pooling(self,x, pool_size=2, stride=2,padding='VALID'):
        return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=stride,padding=padding) 

    def bottleneck_layer(self, x, scope,is_training):
        with tf.name_scope(scope):
            x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch1')#self.training
            x = tf.nn.relu(x)
            x = self.conv_layer(x, filter=128, kernel=1, layer_name=scope+'_conv1')#4 * self.filters=96
            #x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
            x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch2')#self.training
            x = tf.nn.relu(x)
            x = self.conv_layer(x, filter=32, kernel=3, layer_name=scope+'_conv2')#self.filters=24
            #x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
            return x

    def dense_block(self, input_x, nb_layers, layer_name,is_training):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0),is_training=is_training)
            layers_concat.append(x)
            for i in range(nb_layers - 1):
                x = tf.concat(layers_concat, axis=2)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1),is_training=is_training)
                layers_concat.append(x)
            x = tf.concat(layers_concat, axis=2)
            return x
    
    def transition_layer(self, x, scope,is_training):
        with tf.name_scope(scope):
            x = self.Batch_Normalization(x, training=is_training, scope=scope+'_batch1')#self.training
            x = tf.nn.relu(x)
            shape = x.get_shape().as_list()
            in_channel = shape[2]#3
            x = self.conv_layer(x, filter=in_channel*0.5, kernel=1, layer_name=scope+'_conv1')
            #x = self.Drop_out(x, rate=dropout_rate, training=is_training)#self.training
            x = self.Average_pooling(x, pool_size=2, stride=2)
            return x

    def BuildModel(self,
                   DIM = 256,
                   kernel_size = 3
                   ):
        tf.reset_default_graph()
        self.DIM = DIM
        self.kernel_size = kernel_size
        gpu_options = tf.GPUOptions(allow_growth=True)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config=tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7   # 最大显存占用率70%
        config.allow_soft_placement = True 

        """Model"""          
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.training = tf.placeholder(tf.bool)#
            self.seqInput = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
            self.score = self.PredictorNet(self.seqInput,self.training) #self.score_EC,self.score_PA 
            #1.
            #self.label = tf.placeholder(tf.float32, shape=[None,2],name='label')
            #self.BS_y = tf.argmax(self.score_BS, axis=-1,name='label_predict_BS') 
            #self.EC_y = tf.split(self.score,2,axis=1)[0] #tf.argmax(self.score_EC, axis=-1,name='label_predict_EC') 
            #self.PA_y = tf.split(self.score,2,axis=1)[1] #tf.argmax(self.score_PA, axis=-1,name='label_predict_PA') 
            self.label = tf.placeholder(tf.float32, shape=[None,1],name='label')
            self.w = tf.placeholder(tf.float32,shape=[None],name='w')

            """Loss"""
            #2.
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=self.graph)
            self.sess = tf.Session(config=config,graph=self.graph)
            self.loss = tf.losses.mean_squared_error(self.label,self.score) #(?,),(?,1)
            self.saver = tf.train.Saver(max_to_keep=50)
            self.sess.run(tf.initialize_all_variables())
        return



    def get_cls_accuracy0(self):
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

    def get_cls_accuracy(self):
        acc = self.pearsonr_tf(self.score,self.label)#模型预测输出是score, 真实是label
        return acc

    def load_dataset(self,
                   train_data,
                   val_data=None,
                   log_dir='./',
                   log2option=False):
        seq_exp=np.load(train_data,allow_pickle=True)
        self.charmap={'A':0,'G':1,'C':2,'T':3}
        seq_for_merge=[]
        onehot = []
        label = []
        exp_all=[]
        EC_norm=[]
        PA_norm=[]
        seqs=[]
        EC_repeat =[]
        PA_repeat=[]
        Nresample=15
        if log2option:
            for i in range(len(seq_exp)):
                eachexp = np.log2(float(seq_exp[i][1]))
                if eachexp>0 and eachexp<4: #float(seq_exp[i][1])>0 and float(seq_exp[i][1])<4:
                    EC_norm.append(np.array(eachexp))
                    seqs.append(seq_exp[i][0])
        else:
            for i in range(len(seq_exp)):
                if float(seq_exp[i][1])>0 and float(seq_exp[i][1])<4:
                    EC_norm.append(np.array(seq_exp[i][1]))
                    seqs.append(seq_exp[i][0])            
        EC_norm2 = np.array(EC_norm).astype('float32')
        print('after filter min:')
        print(np.min(EC_norm2))

        for i in range(len(seqs)):#seq_exp
            seq=seqs[i]
            exp=EC_norm2[i]
            seq_for_merge.append(seq)
            eachseq = np.zeros([len(seq),4],dtype = 'float')
            for j in range(len(seq)):
                base=seq[j]
                eachseq[j,self.charmap[base]] = 1
            onehot.append(eachseq)
            label.append(np.array(exp))
        self.x = np.array(onehot) #self.seq_list
        self.y = np.array(label).astype('float32') #np.squeeze(label) #self.label_list (10185,3)
        print('after filter: y')
        print(self.y)
        print(np.min(self.y))
        seq_for_merge = np.array(seq_for_merge)
        self.y=np.reshape(self.y,(self.y.shape[0],1))
        self.SEQ_LEN = self.x.shape[1]
        self.c_dim = self.x.shape[2]
        #测试集与训练集划分
        if val_data != None:
            exps_val=[]
            seqs_val=[]
            onehot_val=[]
            label_val=[]
            seq_exp_val=np.load(val_data,allow_pickle=True)
            for i in range(len(seq_exp_val)):
                if float(seq_exp_val[i][1])>0 and float(seq_exp_val[i][1])<4:
                    exps_val.append(np.array(seq_exp_val[i][1]))#seq_exp[i][1][0]
                    seqs_val.append(seq_exp_val[i][0])
            exps_val=np.array(exps_val).astype('float64')

            for i in range(len(seqs_val)):
                seq_val=seqs_val[i]
                exp_val=exps_val[i]
                eachseq_val = np.zeros([len(seq_val),4],dtype = 'float')
                for j in range(len(seq_val)):
                    base=seq_val[j] #seq->seq_val
                    eachseq_val[j,self.charmap[base]] = 1
                onehot_val.append(eachseq_val)
                label_val.append(np.array(exp_val))
            self.val_x = np.array(onehot_val) 
            self.val_y = np.array(label_val).astype('float32')
            self.val_y=np.reshape(self.val_y,(self.val_y.shape[0],1))
        else:
            #d = self.x.shape[0]//10 *9
            #5.10去掉的
            np.random.seed(21)#self.Nseed
            seq_index_A = np.arange(self.x.shape[0])
            np.random.shuffle(seq_index_A)
            n = self.x.shape[0]*int(0.8*10)//10 #
            self.val_x, self.val_y = self.x[seq_index_A[n:],:,:], self.y[seq_index_A[n:],:]#,:#--2.[d:,:]
            self.seq_for_merge = seq_for_merge[seq_index_A[n:]]
            self.x, self.y = self.x[seq_index_A[:n],:,:], self.y[seq_index_A[:n],:]#,:
            #
        

        self.dataset_num = self.x.shape[0]
        self.dataset_num_val =self.val_x.shape[0]
        #输入的表达量分布
        pdf = PdfPages(os.path.join(log_dir,'Input_exp_distribution_y_9.19.pdf'))
        plt.figure(11)
        #sns.distplot(self.y)
        plt.hist(self.y)
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
        return

    def Encoder_rep_beforelastlinear(self,seq,datatype='str'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        out_rep=self.sess.run(self.out_rep,feed_dict={self.seqInput:seq})#
        return out_rep  
        

    
    def Train(self,
              spe='EC',
              lr=0.005,#1e-4,
              beta1=0.5,
              beta2=0.9,
              epoch=1000,
              earlystop=10,
              batch_size=32,
              checkpoint_dir='./predict_model',
              model_name='cls_alone'
              ):
        with self.graph.as_default():
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
            #self.lr = tf.train.exponential_decay(0.002, self.global_step,5,0.96,staircase=True)#5个epoch

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
                    _,loss,cls_accuracy_train= self.sess.run([self.opt,self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.x[I,:,:],self.label:self.y[I,:],self.training:True,self.global_step:epoch})#self.label:self.y[I,:]
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
                loss_val,cls_accuracy_val = self.sess.run([self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.val_x[I_val,:,:],self.label:self.val_y[I_val,:],self.training:False})#self.label:self.y[I,:]
                #loss_val,cls_accuracy_val = self.sess.run([self.loss,self.cls_accuracy],feed_dict={self.seqInput:self.val_x,self.label:self.val_y,self.training:False})
                #cls_accuracy_val_EC=cls_accuracy_val[0]
                #cls_accuracy_val_PA=cls_accuracy_val[1]
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
                    print('cls_acc loss:'+str(best_loss)+'(epoch:'+str(epoch)+')'+',now highest acc:'+str(best_percent))
                    self.save(self.checkpoint_dir,epoch,'bestloss',cls_accuracy_val,spe)
                else:
                    convIter += 1
                    if convIter>=earlystop:
                        print('best acc:'+str(best_percent))#+'best acc(PA):'+str(best_percent[1]
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

    def save(self, checkpoint_dir, step,name='pt',acc=[0,0],spe='EC'):
        # with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
        #     for c in self.charmap:
        #         f.write(c+'\t')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        checkpoint_dir = os.path.join(checkpoint_dir,spe)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir,name+'_acc_'+str(acc)+'_'+self.model_name +'_epoch_'+str(step)+'_'+ '.model'), global_step=step)#checkpoint_dir, self.model_name + '.model'
        


    def load(self, checkpoint_dir= None, model_name = None,log_dir=None,spe=None,fign=None):
        print(" [*] Reading checkpoints...")
        if checkpoint_dir == None:
            checkpoint_dir = self.checkpoint_dir
        if model_name == None:
            model_name = self.model_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        checkpoint_dir = os.path.join(checkpoint_dir,spe)
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
            #plot_ECorPA_list=['EC','PA']
            #color_list='red','blue']
            xlimECPA = {'EC':4,'PA':2}
            #datadjx = pd.DataFrame({'xdjx': [-2,5], 'ydjx': [-2,5]})#-3,9
            #for plot_ECorPA in range(len(plot_ECorPA_list)):
                #print('plot_ECorPA:')
                #print(plot_ECorPA)
            #spe = plot_ECorPA_list[plot_ECorPA]
            #Val:
            data_val = pd.DataFrame(columns=['Promoter activity by MPRA','Predicted promoter activity'])
            data_val['Predicted promoter activity']=val_pred.reshape(-1,)#list(np.split(val_pred,2,axis=1)[plot_ECorPA].reshape(-1,)) #np.split(y,2,axis=1)[0]
            data_val['Promoter activity by MPRA']=self.val_y.reshape(-1,)#list(np.split(self.val_y,2,axis=1)[plot_ECorPA].reshape(-1,))

            val_R_PCC = pearsonr(data_val['Promoter activity by MPRA'],data_val['Predicted promoter activity'])[0]
            val_R_spearman = spearmanr(data_val['Promoter activity by MPRA'],data_val['Predicted promoter activity'])[0]
            print(spe+': val_R_PCC='+str(val_R_PCC))
            print(spe+': val_R_spearman='+str(val_R_spearman))
            pdf = PdfPages(os.path.join(log_dir,'denselstm_true_pred_scatter_val_'+spe+'.pdf'))
            plt.figure(int(fign),figsize=(10, 10))
            sns.set(style="whitegrid",font_scale=1.2)
            # g=sns.regplot(x='Promoter activity by MPRA', y='Predicted promoter activity', data=data_val,
            #         color='#000000',
            #         marker='.',
            #         scatter_kws={'s': 5,'color':'brown'},#设置散点属性，参考plt.scatter
            #         line_kws={'linestyle':'--','color':'darkgrey'})#设置线属性，参考 plt.plot 
            real_expr= np.array(data_val['Promoter activity by MPRA'])
            predict_expr=np.array(data_val['Predicted promoter activity']) 
            plt.title('pearson coefficient(train):{:.2f}'.format(val_R_PCC))
            plt.scatter(real_expr,predict_expr, alpha=0.5, c='brown')
            regressor = LinearRegression()
            regressor = regressor.fit(np.reshape(real_expr,(-1, 1)),np.reshape(predict_expr,(-1, 1)))
            plt.plot(np.reshape(real_expr,(-1,1)), regressor.predict(np.reshape(real_expr,(-1,1))),c='darkgrey',linestyle='dashed',linewidth=2)
            plt.xlabel('Observed activity (log2)',fontsize=11)
            plt.ylabel('Predicted expression (log2)',fontsize=11)
            plt.xlim((0,4))#(-3,4)
            plt.ylim((0,4))#(-3,4)
            #plt.xlim(-1,xlimECPA[spe]) #EC:-1,4; PA:-1,2
            #plt.ylim(-1,xlimECPA[spe])
            pdf.savefig()
            pdf.close()

            #Train:
            data_train = pd.DataFrame(columns=['Promoter activity by MPRA','Predicted promoter activity'])
            data_train['Predicted promoter activity']=list(train_pred.reshape(-1,))#data_train['Predicted promoter activity']=list(np.split(train_pred,2,axis=1)[plot_ECorPA].reshape(-1,))
            data_train['Promoter activity by MPRA']=list(self.y.reshape(-1,))#data_train['Promoter activity by MPRA'] = list(np.split(self.y,2,axis=1)[plot_ECorPA].reshape(-1,))
            train_R_PCC = pearsonr(data_train['Promoter activity by MPRA'],data_train['Predicted promoter activity'])[0]
            print(spe+': train_R='+str(train_R_PCC))
            pdf = PdfPages(os.path.join(log_dir,'denselstm_true_pred_scatter_train_'+spe+'.pdf'))
            plt.figure(int(6),figsize=(10, 10))
            # g=sns.regplot(x='Promoter activity by MPRA', y='Predicted promoter activity', data=data_train,
            #         color='#000000',
            #         marker='+',
            #         scatter_kws={'s': 40,'color':'grey',},#设置散点属性，参考plt.scatter
            #         line_kws={'linestyle':'--','color':'r'})#设置线属性，参考 plt.plot 
            plt.title('pearson coefficient(train):{:.2f}'.format(train_R_PCC))
            plt.scatter(data_train['Promoter activity by MPRA'],data_train['Predicted promoter activity'], alpha=0.5, c='brown')
            plt.xlabel('Observed activity (log2)',fontsize=11)
            plt.ylabel('Predicted expression (log2)',fontsize=11)
            plt.xlim((0,5))#(-3,4)
            plt.ylim((0,5))#(-3,4)
            pdf.savefig()
            pdf.close()
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
            y.append(self.sess.run(self.score,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:],self.training:False}))
        y = np.concatenate(y)
        y = y.flatten()#np.reshape(y,(y.shape[0]))
        # else:
        #     for b in range(batches):
        #         eachy_final=[]
        #         eachy=self.sess.run([self.EC_y,self.PA_y],feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]})
        #         eachy=np.array(eachy) #2,256,1
        #         eachy = np.squeeze(eachy)#2,256 2,251
        #         for i in range(len(eachy[0])):
        #             eachy_final.append([eachy[0][i],eachy[1][i]])
        #         eachy_final = np.array(eachy_final)
        #         y.append(eachy_final)
        #     y = np.concatenate(y)
        return y
    
    # def Predictor(self,seq,datatype='str',spe='ECandPA'):
    #     if datatype == 'str':
    #         seq = seq2oh(seq,self.charmap)
    #     num = seq.shape[0]
    #     batches = math.ceil(num/self.BATCH_SIZE)
    #     y = []
    #     if spe =='EC':
    #         for b in range(batches):
    #             y.append(self.sess.run(self.EC_y,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
    #         y = np.concatenate(y)
    #         y = np.reshape(y,(y.shape[0]))
    #     elif spe =='PA':
    #         for b in range(batches):
    #             y.append(self.sess.run(self.PA_y,feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]}))
    #         y = np.concatenate(y)
    #         y = np.reshape(y,(y.shape[0]))
    #     else:
    #         for b in range(batches):
    #             eachy_final=[]
    #             eachy=self.sess.run([self.EC_y,self.PA_y],feed_dict={self.seqInput:seq[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:,:]})
    #             eachy=np.array(eachy) #2,256,1
    #             eachy = np.squeeze(eachy)#2,256 2,251
    #             for i in range(len(eachy[0])):
    #                 eachy_final.append([eachy[0][i],eachy[1][i]])
    #             eachy_final = np.array(eachy_final)
    #             y.append(eachy_final)
    #         y = np.concatenate(y)
    #     return y

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

    def pearsonr_tf1(self,x,y):
        mx = tf.reduce_mean(x)
        my = tf.reduce_mean(y)
        xm, ym = x - mx, y - my
        t1_norm = tf.nn.l2_normalize(xm, axis = 1)
        t2_norm = tf.nn.l2_normalize(ym, axis = 1)
        cosine = tf.losses.cosine_distance(t1_norm, t2_norm, axis = 1)
        return cosine

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

    def promoter_scoring_out(self,promoter_dir,input_filename,out_filename,Nout): #读入读出都是promoter_dir
        seqs,names=self.file2seq(promoter_dir,input_filename)
        filtered_out=open(os.path.join(promoter_dir, out_filename),'w')
        y=self.Predictor(seq=seqs)
        seq_exp=pd.DataFrame(columns=['promoter','predicted_strength','name'])
        seq_exp['promoter']=seqs[:len(y)]
        seq_exp['predicted_strength']=y
        seq_exp['name']=names
        seq_exp=seq_exp.sort_values(by=['predicted_strength'],ascending=False)
        seq_exp = seq_exp.reset_index()
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        #print('seq_exp:')
        #print(seq_exp)
        #print('first_line')
        #
        if Nout<=len(seq_exp):
            count=0
            print('writing out seqs...')
            for i in range(Nout):
                count+=1
                seq = str(seq_exp['promoter'][i])
                seq = 'ACTGGCCGCTTGACG'+seq+'CACTGCGGCTCCTGC' 
                filtered_out.write('>'+seq_exp['name'][i][1:]+'-'+str(count)+'\n')
                #filtered_out.write(str(seq_exp['promoter'][i])+'\n')
                filtered_out.write(seq+'\n')
        else:
            print('not enougth AI_designed promoters,please generate more seqs')
        filtered_out.close()
        print('[Done] Promoters written out.')
        return

    def file2seq(self,dir,filename):
        f=open(os.path.join(dir, filename))
        seqs=[]
        name=[]
        for line in f.readlines():
            line=line.strip('\n')
            if '>' not in line:
                seqs.append(line.strip('\n'))
            else:
                name.append(line.strip('\n'))
        f.close()
        return seqs,name

    def promoter_scoring(self,promoter_dir,input_filename): #读入读出都是promoter_dir
        seqs,names=self.file2seq(promoter_dir,input_filename)
        y=self.Predictor(seq=seqs)
        seq_exp=pd.DataFrame(columns=['promoter','predicted_strength','name'])
        seq_exp['promoter']=seqs[:len(y)]
        seq_exp['predicted_strength']=y
        seq_exp['name']=names
        return seq_exp
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


    def weightMSEloss(x,y,threshold=0.4):#x=真实Label,y=预测label
        w = 2*tf.tanh(x-threshold)+3 # 2*+3#+10 #2*tf.nn.relu(x-threshold) #
        #print('w shape:')
        #print(x.shape[0])
        n= 256 #x.shape[0]
        loss = K.sum(tf.multiply(K.square(x-y),w))/n
        return loss

    def GradientDescent(self, oh):
        oh = np.array(oh)
        oh = oh.astype('float32')
        with self.graph.as_default():
            if hasattr(self,'oh_tensor') == False:
                self.oh_tensor = tf.Variable(oh,name='oh_tensor')
            self.sess.run(tf.assign(self.oh_tensor,oh))
            #gradient = self.sess.run(tf.gradients(self.PredictorNet(self.oh_tensor,reuse=True),[self.oh_tensor])[0])
            gradient = self.sess.run(tf.gradients(self.PredictorNet(self.oh_tensor,reuse=True),[self.oh_tensor]))
            return gradient
            
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
