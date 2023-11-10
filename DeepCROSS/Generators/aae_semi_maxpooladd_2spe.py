import numpy as np
import tensorflow as tf
import time
import pickle
from ..ops import Conv1D, Linear, ResBlock
from keras.layers import MaxPooling1D,UpSampling1D
from keras.layers import Bidirectional,LSTM,Dropout
from ..ops.param import params_with_name
from ..ProcessData import * 
from .kmer_statistics import kmer_statistics
import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')

class AAE_semi_maxpooladd_2spe:
    def __init__(self, 
                 log_dir='./',
                 nbin=3
                 ):
        self.log_dir=log_dir
        self.epoch_count=0
        self.plotbin=False
        self.artifical_rep=[]
        self.artifical_spe=[]
        self.penvalue = 0.1
        self.nbin=nbin
        self.gdN=512
        self.eta = 0.2 
        print('penvalue(ratio): '+str(self.penvalue))
        print('eta(distance threshold): '+str(self.eta))

    def EncoderNet(self, x, is_training=True, reuse=False):
        with tf.variable_scope("Encoder", reuse=reuse):
            output = Conv1D('Conv1D.1', self.c_dim, self.DIM, 6, x)
            output = tf.nn.relu(output)
            output = MaxPooling1D(pool_size=2)(output) 
            output = Conv1D('Conv1D.2', self.DIM, self.DIM*2, 3, output)
            output = tf.nn.relu(output)
            output = MaxPooling1D(pool_size=2)(output)
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM*2, 3, 'ResBlock.{}'.format(i))
            output = Conv1D('Conv_label.1', self.DIM*2, self.DIM*4, 3, output)
            output = tf.nn.relu(output)
            output = Conv1D('Conv_label.2', self.DIM*4, self.DIM*4, 3, output)
            output = tf.nn.relu(output)
            output = Conv1D('Conv_label.3', self.DIM*4, self.DIM*4, 3, output)
            output = tf.nn.relu(output)
            return output 

    def DecoderNet(self, z, apply_softmax=True, is_training=True, reuse=False):
        with tf.variable_scope("Generator", reuse=reuse):
            output = Linear('Dense.1', self.Z_DIM, self.DIM*4*int(self.SEQ_LEN/2/2), z)
            output = tf.reshape(output, [-1, int(self.SEQ_LEN/2/2),self.DIM*4])
            output = Conv1D('Conv.1', self.DIM*4,self.DIM*4, 3, output)
            output = tf.nn.relu(output)
            output = Conv1D('Conv.2', self.DIM*4,self.DIM*4, 3, output)
            output = tf.nn.relu(output)
            output = Conv1D('Conv.3', self.DIM*4,self.DIM*2, 3, output)
            output = tf.nn.relu(output)
            for i in range(1,1+self.n_layers):
                output = ResBlock(output, self.DIM*2, 3, 'ResBlock.decoder.{}'.format(i)) 
            output = UpSampling1D(2)(output) 
            output = Conv1D('Conv.4',self.DIM*2,self.DIM, 3, output) 
            output = UpSampling1D(2)(output) 
            output = Conv1D('Conv.5', self.DIM, self.c_dim, 6, output) 
            output = tf.reshape(output, [-1, 4*int(self.SEQ_LEN-1)])
            output = Linear('Dense.2', 4*int(self.SEQ_LEN-1), 4*int(self.SEQ_LEN), output)
            output = tf.reshape(output, [-1, int(self.SEQ_LEN),4])
            if apply_softmax == True:
                output = tf.nn.softmax(output)
            return output
    
    def cls_layer(self, encoder_out):
        with tf.variable_scope("cls_layer"):
            output = Dropout(0.2)(encoder_out) 
            output = Bidirectional(LSTM(50,return_sequences=True))(output) 
            output = Linear('Dense_label.1',41*50*2,self.gdN, output)
            output = Dropout(0.2)(output)
            cls_logits_EC =  Linear('Dense_label.3',self.gdN,self.nbin, output)
            cls_logits_PA =  Linear('Dense_label.4',self.gdN,self.nbin, output)
            return cls_logits_EC,cls_logits_PA

    def cluster_layer(self,out_oh_softmax,Wc):
        with tf.variable_scope("cluster"):
            cluster_head_out=tf.matmul(out_oh_softmax,Wc)
        return cluster_head_out

    def linear_before_z(self,afterWc_concat):
        with tf.variable_scope('linear_before_z'):
            out= Linear('Dense_before_z',self.m*3,self.Z_DIM, afterWc_concat)
            return out
    
    def linear_z_layer(self,encoderout):
        with tf.variable_scope("linear_z"):
            output = tf.reshape(encoderout, [-1, self.DIM*4*int(self.SEQ_LEN/2/2)])
            output= Linear('Dense_z.1', self.DIM*4*int(self.SEQ_LEN/2/2),1024, output)
            out= Linear('Dense_z.2', 1024,self.Z_DIM, output)
        return out

    def DiscriminatorNet_z(self, z, is_training=True, reuse=False):
        with tf.variable_scope("Discriminator_z", reuse=reuse):
            output = Linear('Dense.z.1',self.Z_DIM, 64*self.Z_DIM,z)
            output = tf.nn.leaky_relu(output)
            output = Linear('output_z',64*self.Z_DIM, 1,output)
        return output

    def DiscriminatorNet_y_EC(self,y,is_training=True,reuse=False): 
        with tf.variable_scope("Discriminator_y_EC"):
            output = Linear('Dense.y.1.ec',self.nbin, 1024,y)
            output = tf.nn.leaky_relu(output)
            out = Linear('output_y.ec',1024, 1,output)
        return out

    def DiscriminatorNet_y_PA(self,y,is_training=True,reuse=False):
        with tf.variable_scope("Discriminator_y_PA"): 
            output = Linear('Dense.y.1.pa',self.nbin, 1024,y)
            output = tf.nn.leaky_relu(output)
            out = Linear('output_y.pa',1024, 1,output)
        return out
    
    def Encoder_z(self,seq,datatype='str'):
        if datatype == 'str':
            seq = seq2oh(seq,self.charmap)
        z=self.sess.run(self.gen_z,feed_dict={self.real_input:seq})
        return z

    def Generate_rep(self,label,generated_z,gen_batch_size):
        self.generated_z =tf.placeholder(tf.float32, shape=[None, self.Z_DIM],name='generated_z')
        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            label = tf.convert_to_tensor(label)
            label_EC,label_PA=tf.split(label,num_or_size_splits=2, axis=1)
            one_hot_label_EC = tf.reshape(tf.one_hot(label_EC, self.nbin),(gen_batch_size,self.nbin)) 
            one_hot_label_PA = tf.reshape(tf.one_hot(label_PA, self.nbin),(gen_batch_size,self.nbin))  
            Wc_EC = tf.get_variable("Wc_EC_name")
            Wc_PA = tf.get_variable("Wc_PA_name")
            clusterhead_EC = self.cluster_layer(one_hot_label_EC,Wc_EC)
            clusterhead_PA = self.cluster_layer(one_hot_label_PA,Wc_PA)
            clusterhead = tf.concat((clusterhead_EC,clusterhead_PA),axis=-1)
            self.rep_out = tf.add(clusterhead,self.generated_z)
        rep_out=self.sess.run(self.rep_out,feed_dict={self.generated_z:generated_z})
        return rep_out

        
    def Generator_seq(self,label_gen=None,batchN=10,z=None): 
        if z is None:
            z = np.random.normal(size=(self.BATCH_SIZE*batchN,self.Z_DIM))
        if label_gen==None:
            label_gen = np.concatenate([np.random.choice(self.nbin, 1),np.random.choice(self.nbin, 1)])
        num = z.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)
        generated_seq = []
        rep_in_report_10batch_lastsaveEpoch=[]
        for b in range(batches-1):
            rep_in_report=self.sess.run(self.rep_in,feed_dict={self.random_z:z[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:],self.bin_gen:label_gen})
            oh=self.sess.run(self.gen_oh,feed_dict={self.rep_in:rep_in_report})
            generated_seq.extend(oh2seq(oh,self.invcharmap))
            rep_in_report_10batch_lastsaveEpoch.extend(rep_in_report)
        return rep_in_report_10batch_lastsaveEpoch,generated_seq


    def Generator_forfilter(self,Ntime=10,z=None,seed=1,n=1,label_gen=[0,2],filename='EConly'):
        if z is None:
            np.random.seed(seed)
            z = np.random.normal(size=(self.BATCH_SIZE*Ntime,self.Z_DIM))
        generated_seq = []
        num = z.shape[0]
        batches = math.ceil(num/self.BATCH_SIZE)-1
        for b in range(batches):
            rep_in_report=self.sess.run(self.rep_in,feed_dict={self.random_z:z[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:],self.bin_gen:label_gen})
            oh=self.sess.run(self.gen_oh,feed_dict={self.rep_in:rep_in_report})
            generated_seq.extend(oh2seq(oh,self.invcharmap))
        with open(self.log_dir+'/genseq_onebillion_divide_'+filename, 'ab') as f:
                    pickle.dump(generated_seq, f) 
        return 

    def Generate_rep_seq_getanybin(self,label_gen=[2,2],batchN=10):
        artifical_rep=self.Generator_seq(label_gen=label_gen,batchN=batchN)[0] 
        artifical_spe=['AAE_generated']*len(self.artifical_rep)
        with open(self.log_dir+'/AAE_generated_'+str(label_gen[0])+'_'+str(label_gen[1])+'_rep', 'wb') as f:
            pickle.dump(self.artifical_rep, f) 
        with open(self.log_dir+'/AAE_generated'+str(label_gen[0])+'_'+str(label_gen[1])+'_label', 'wb') as f:
            pickle.dump(self.artifical_spe, f) 
        batches = math.ceil(len(artifical_rep)/self.BATCH_SIZE)
        generated_seq = []
        for b in range(batches-1):
            oh=self.sess.run(self.gen_oh,feed_dict={self.rep_in:artifical_rep[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE]})
            generated_seq.extend(oh2seq(oh,self.invcharmap))
        saveseq(self.log_dir + '/designed_sequences/'+'generated_seq_'+str(label_gen[0])+'_'+str(label_gen[1])+'.txt',generated_seq)
        return generated_seq

    def xent_loss(self, inn, outt,cluster_loss_in):
        out_recon=tf.reduce_mean(tf.losses.softmax_cross_entropy(inn,outt))
        out = out_recon+cluster_loss_in
        return out
    
    def _get_cluster_loss(self,Wc):
        cluster_loss=0 
        for i in range(Wc.get_shape()[0]):
            each_row=Wc[i,:]
            for j in range(i+1,Wc.get_shape()[0]):
                dist_two=tf.square(Wc[i,:]-Wc[j,:])
                eachloss=tf.nn.relu(self.eta-tf.reduce_mean(dist_two))*self.penvalue
                cluster_loss+=eachloss
        out=tf.reduce_mean(cluster_loss)
        return out 

    def get_cluster_loss(self,Wc_EC,Wc_PA):
        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            cluster_loss_all=tf.add_n([self._get_cluster_loss(Wc_EC),self._get_cluster_loss(Wc_PA)])/3
            return cluster_loss_all

    def gradient_penalty(self, real, fake,net):
        fake = tf.dtypes.cast(fake, tf.float32)
        real = tf.dtypes.cast(real, tf.float32)
        differences = fake - real
        alpha_shape=real.get_shape()
        alpha = tf.random_uniform(
                shape=[self.BATCH_SIZE,1,1], 
                minval=0.,
                maxval=1.
                )
        interpolates = real + (alpha*differences)
        gradients = tf.gradients(net(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty
    
    def discriminator_loss(self, real, fake):
        return -tf.reduce_mean(real) + tf.reduce_mean(fake)
    
    def encoder_loss(self,fake):
        return -tf.reduce_mean(fake)
    
    def BuildModel(self,
                   datafile,
                   univ_file,
                   univ_val_file,
                   kernel_size=3,
                   Z_DIM=64,
                   DIM=128,
                   n_layers=8,
                   supervise_file='EC_PA_exp_bin_3_train80_originnobin.npy',
                   BATCH_SIZE=64,
                   LAMBDA=10
                   ):
        print('loading dataset...')
        self.data,self.charmap,self.invcharmap,self.inputseqs = load_seq_data_y(datafile,labelflag=1) 
        self.univ_data,self.univ_charmap,self.univ_invcharmap,_ = load_seq_data_y(univ_file)
        self.univ_data_val,self.univ_charmap_val,self.univ_invcharmap_val,_ = load_seq_data_y(univ_val_file)
        np.random.seed(3)
        seq_index_A = np.arange(self.data.shape[0])
        np.random.shuffle(seq_index_A)
        n = self.data.shape[0]*9//10
        n_train=seq_index_A[:n]
        n_val=seq_index_A[n:]
        print(np.max(seq_index_A))
        self.data_val = self.data[n_val,:,:]
        self.data = self.data[n_train,:,:]
        self.dataset_num = self.data.shape[0]
        self.SEQ_LEN = self.data.shape[1]
        self.c_dim = self.data.shape[2]
        self.kernel_size = kernel_size
        self.DIM = DIM
        self.Z_DIM = Z_DIM
        self.n_layers = n_layers
        self.BATCH_SIZE = BATCH_SIZE
        self.supervise_file=supervise_file
        self.label_data = self.read_train_data(self.BATCH_SIZE,nbin=self.nbin,supervise_file=self.supervise_file,batch_dict_name=['seq_train', 'label_train','seq_valid','label_valid'])
        self.LAMBDA = LAMBDA
        self.layers = {}
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('Building model...')
        self.real_input = tf.placeholder(tf.float32, shape=[None, self.SEQ_LEN, self.c_dim],name='input')
        self.label = tf.placeholder(tf.float32, name='label', shape=[None,2])
        self.real_y = tf.placeholder(tf.int64, name='real_y', shape=[None,2]) 
        self.rep = tf.placeholder(tf.float32, name='rep', shape=[None,self.Z_DIM]) 
        self.m=32
        self.bin_gen=tf.placeholder(tf.int64, name='bingen', shape=[2]) 
        """ Concate Model """
        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            self.Wc_EC = tf.get_variable('Wc_EC_name',shape=[self.nbin,self.m],initializer=tf.constant_initializer(0.4))
            self.Wc_PA = tf.get_variable('Wc_PA_name',shape=[self.nbin,self.m],initializer=tf.constant_initializer(0.4))
            self.layers['encoder_out']=self.EncoderNet(self.real_input)
            self.layers['z']= self.linear_z_layer(self.layers['encoder_out'])
            self.layers['cls_logits'] = self.cls_layer(self.layers['encoder_out'])
            self.layers['cls_logits_EC'],self.layers['cls_logits_PA']=self.cls_layer(self.layers['encoder_out'])
            self.layers['y_EC'] = tf.argmax(self.layers['cls_logits_EC'], axis=-1,name='label_predict_EC')
            self.layers['y_PA'] = tf.argmax(self.layers['cls_logits_PA'], axis=-1,name='label_predict_PA')
            self.layers['one_hot_y_approx_EC'] = tf.nn.softmax(self.layers['cls_logits_EC'], axis=-1)
            self.layers['one_hot_y_approx_PA'] = tf.nn.softmax(self.layers['cls_logits_PA'], axis=-1)
            self.layers['cluster_head_EC'] =  self.cluster_layer(self.layers['one_hot_y_approx_EC'],self.Wc_EC)
            self.layers['cluster_head_PA'] =  self.cluster_layer(self.layers['one_hot_y_approx_PA'],self.Wc_PA)
            self.layers['cluster_head'] = tf.concat((self.layers['cluster_head_EC'],self.layers['cluster_head_PA']), axis=-1)
            self.layers['rep'] = tf.add(self.layers['z'],self.layers['cluster_head'])
            out_logits = self.DecoderNet(self.layers['rep'],apply_softmax=False)
            self.reconstruct_seqs = tf.argmax(out_logits, axis=-1,name='reconstruct_seq')
            self.cluster_loss = self.get_cluster_loss(self.Wc_EC,self.Wc_PA)
            self.g_loss = self.xent_loss(self.real_input,out_logits,self.cluster_loss)
        with tf.variable_scope('regularization_z'):
            z_samples = tf.random_normal(shape=(self.BATCH_SIZE,self.Z_DIM))
            real_logits_z = self.DiscriminatorNet_z(z_samples)
            fake_logits_z = self.DiscriminatorNet_z(self.layers['z'],reuse=True)
            GP_z = self.gradient_penalty(z_samples,self.layers['z'],self.DiscriminatorNet_z)
            self.d_loss_z = self.discriminator_loss(real_logits_z,fake_logits_z) + self.LAMBDA*GP_z
        with tf.variable_scope('regularization_y'):
            self.real_in_y = tf.one_hot(self.real_y, self.nbin) 
            self.real_in_y_EC,self.real_in_y_PA=tf.split(self.real_in_y,num_or_size_splits=2, axis=1)
            self.real_in_y_EC=tf.reshape(self.real_in_y_EC,(-1,self.nbin))
            self.real_in_y_PA=tf.reshape(self.real_in_y_PA,(-1,self.nbin))
            real_logits_y_EC = self.DiscriminatorNet_y_EC(self.real_in_y_EC)
            real_logits_y_PA = self.DiscriminatorNet_y_PA(self.real_in_y_PA)
            fake_logits_y_EC = self.DiscriminatorNet_y_EC(self.layers['one_hot_y_approx_EC'],reuse=True)
            GP_y_EC = self.gradient_penalty(self.real_in_y_EC,self.layers['one_hot_y_approx_EC'],self.DiscriminatorNet_y_EC)
            self.d_loss_y_EC = self.discriminator_loss(real_logits_y_EC,fake_logits_y_EC) + self.LAMBDA*GP_y_EC
            fake_logits_y_PA = self.DiscriminatorNet_y_PA(self.layers['one_hot_y_approx_PA'],reuse=True)
            GP_y_PA = self.gradient_penalty(self.real_in_y_PA,self.layers['one_hot_y_approx_PA'],self.DiscriminatorNet_y_PA) 
            self.d_loss_y_PA = self.discriminator_loss(real_logits_y_PA,fake_logits_y_PA) + self.LAMBDA*GP_y_PA 
            self.d_loss_y = tf.add_n([self.d_loss_y_EC+self.d_loss_y_PA])/2
        self.e_loss_z = self.encoder_loss(fake_logits_z)
        self.e_loss_y = self.encoder_loss(tf.add_n([fake_logits_y_EC,fake_logits_y_PA])/2)
        self.cls_loss = self.get_cls_loss()

        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            self.gen_z = self.linear_z_layer(self.EncoderNet(self.real_input,reuse=True))
            self.random_z = tf.placeholder(tf.float32, shape=[None, self.Z_DIM],name='random_z')
            label_gen_EC = []
            label_gen_PA = []
            for k in range(self.BATCH_SIZE):
                label_gen_EC.extend([self.bin_gen[0]])
                one_hot_label_EC = tf.one_hot(label_gen_EC, self.nbin)
            for k in range(self.BATCH_SIZE):
                label_gen_PA.extend([self.bin_gen[1]])
                one_hot_label_PA = tf.one_hot(label_gen_PA, self.nbin)
            Wc_EC = tf.get_variable("Wc_EC_name")
            Wc_PA = tf.get_variable("Wc_PA_name")
            clusterhead_EC = self.cluster_layer(one_hot_label_EC,Wc_EC)
            clusterhead_PA = self.cluster_layer(one_hot_label_PA,Wc_PA)
            clusterhead = tf.concat((clusterhead_EC,clusterhead_PA),axis=-1)
            self.rep_in = tf.add(clusterhead,self.random_z)
            self.gen_oh = self.DecoderNet(self.rep_in,reuse=True,is_training=False) 
        """ Summary """
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_sum_y = tf.summary.scalar("d_loss_y", self.d_loss_y)
        self.d_sum_z = tf.summary.scalar("d_loss_z", self.d_loss_z)
        self.e_sum_y = tf.summary.scalar("e_loss_y", self.e_loss_y)
        self.e_sum_z = tf.summary.scalar("e_loss_z", self.e_loss_z)
        self.cls_sum = tf.summary.scalar("cls_loss", self.cls_loss)
        self.saver = tf.train.Saver(max_to_keep=1)


    def Train(self,
              learning_rate=1e-4,
              beta1=0.5,
              beta2=0.9,
              save_freq = 150,
              supervise_freq =50,
              epoch=1000,
              sample_dir='./samples',
              checkpoint_dir = './generative_model',
              model_name='aae',
              log_dir = './log'):
        self.iteration = self.dataset_num // self.BATCH_SIZE
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_freq = min(save_freq, self.iteration)
        self.supervise_freq = min(supervise_freq,self.iteration)
        self.epoch = epoch
        self.sample_dir = sample_dir
        if os.path.exists(self.sample_dir) == False:
            os.makedirs(self.sample_dir)
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir) == False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = model_name
        self.log_dir = log_dir
        if os.path.exists(self.log_dir) == False:
            os.makedirs(self.log_dir)
        self._cls_accuracy_op = self.get_cls_accuracy()
        self._get_label = self.get_label()
        self._get_predict_label = self.get_predict_label()
        self.plotfreq=3
        t_vars = params_with_name('')
        e_vars_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/Encoder') +tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/linear_z')
        e_vars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/Encoder') +tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/cls_layer')
        d_vars_z = [var for var in t_vars if 'regularization_z' in var.name]
        d_vars_y = [var for var in t_vars if 'regularization_y' in var.name]
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE')

        self.cls_learning_rate=self.learning_rate
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.e_opt_y = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.e_loss_y, var_list=e_vars_y)
            self.e_opt_z = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.e_loss_z, var_list=e_vars_z)
            self.d_opt_y = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss_y, var_list=d_vars_y)
            self.d_opt_z = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss_z, var_list=d_vars_z)
            self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
            self.cls_opt = tf.train.AdamOptimizer(self.cls_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.cls_loss, var_list=e_vars_y)

        self.sess.run(tf.initialize_all_variables())

        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        true_kmer = [kmer_statistics(i, oh2seq(self.data,self.invcharmap)) for i in [4,6,8]]
        val_kmer = [kmer_statistics(i, oh2seq(self.data_val,self.invcharmap)) for i in [4,6,8]]
        val_js_val_true = [val_kmer[i].js_with(true_kmer[i]) for i in range(3)]
        print('js_valid: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(val_js_val_true[0],val_js_val_true[1],val_js_val_true[2]))
        true_kmer_univ = [kmer_statistics(i, oh2seq(self.univ_data,self.univ_invcharmap)) for i in [4,6,8]]
        val_kmer_univ = [kmer_statistics(i, oh2seq(self.univ_data_val,self.univ_invcharmap_val)) for i in [4,6,8]]
        val_js_val_true_univ = [val_kmer_univ[i].js_with(true_kmer_univ[i]) for i in range(3)]
        print('js_valid_univ: js_4mer_univ: {}, js_6mer_univ: {}, js_8mer_univ: {}'.format(val_js_val_true_univ[0],val_js_val_true_univ[1],val_js_val_true_univ[2]))

        true_kmer = [kmer_statistics(i, oh2seq(self.data,self.invcharmap)) for i in [4,6,8]]
        gen = self.inf_train_gen()
        counter = 1
        flag=0
        start_time = time.time()
        conv = 0
        best_js = 1
        train_6js = []
        val_6js = []
        train_6js_univ = []
        val_6js_univ =[]
        cls_accuracy_sum = 0
        cls_accuracy_valid_all=[]
        cls_accuracy_train_all=[]
        for epoch in range(1, 1+self.epoch):
            for idx in range(1, 1+self.iteration):
                y_real_sample_EC = np.random.choice(self.nbin, self.BATCH_SIZE).reshape(-1,1)
                y_real_sample_PA = np.random.choice(self.nbin, self.BATCH_SIZE).reshape(-1,1)
                y_samples = np.concatenate([y_real_sample_EC,y_real_sample_PA],axis=1)
                

                for i in range(3):
                    _data = gen.__next__()
                    _, d_loss_z,summary_str1 = self.sess.run([self.d_opt_z,self.d_loss_z,self.d_sum_z],feed_dict={self.real_input:_data})
                    _data = gen.__next__()
                    _, d_loss_y,summary_str2 = self.sess.run([self.d_opt_y,self.d_loss_y,self.d_sum_y],feed_dict={self.real_input:_data,self.real_y:y_samples})
                self.writer.add_summary(summary_str1, counter)
                self.writer.add_summary(summary_str2, counter)
                _data = gen.__next__()
                _, e_loss_y,summary_str3 = self.sess.run([self.e_opt_y,self.e_loss_y,self.e_sum_y],feed_dict={self.real_input:_data})
                _data = gen.__next__()
                _, e_loss_z,summary_str4 = self.sess.run([self.e_opt_z,self.e_loss_z,self.e_sum_z],feed_dict={self.real_input:_data})
                
                self.writer.add_summary(summary_str3, counter)
                self.writer.add_summary(summary_str4, counter)
                for i in range(3):
                    _data = gen.__next__()
                    _,reconstruct_seq_oh,g_loss,cluster_loss,summary_str5 = self.sess.run([self.g_opt,self.reconstruct_seqs,self.g_loss,self.cluster_loss,self.g_sum],feed_dict={self.real_input:_data})
                
                self.writer.add_summary(summary_str5, counter)

                counter += 1
                hamming_distance=self.cal_hamming(_data,reconstruct_seq_oh)
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, e_loss_z: %.8f,e_loss_y: %.8f, d_loss_z: %.8f,d_loss_y: %.8f, g_loss: %.8f, cluster_loss: %.8f, reconstruction hamming distance:%.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, e_loss_z,e_loss_y, d_loss_z, d_loss_y,g_loss,cluster_loss,hamming_distance))
                if np.mod(idx, self.save_freq) == 0:
                    gen_univ_seq=[]
                    self.artifical_rep=[]
                    self.artifical_spe=[]
                    exp_bin_eachepoch=[2,2]
                    gen_univ_seq.extend(self.Generator_seq(label_gen=exp_bin_eachepoch,batchN=40)[1]) 
                    self.artifical_rep.extend(self.Generator_seq(label_gen=exp_bin_eachepoch,batchN=40)[0])
                    self.artifical_spe.extend(['AAE_generated']*len(self.artifical_rep))
                    saveseq(self.sample_dir + '/' + self.model_name + 
                                '_train_222_{:02d}_{:05d}.txt'.format(epoch, idx + 1)
                                ,gen_univ_seq)
                    
                if np.mod(idx,self.supervise_freq) == 0 and epoch>1:
                    batch_data = self.label_data.next_batch_dict(labeluse=True)
                    seq_train = batch_data['seq_train']
                    label_train = batch_data['label_train']
                    seq_valid = batch_data['seq_valid']
                    label_valid = batch_data['label_valid']
                    _, cls_loss_value, cls_accuracy_train,labels_onehot,predict_logits,summary_str5= self.sess.run([self.cls_opt,self.cls_loss,self._cls_accuracy_op,self._get_label,self._get_predict_label,self.cls_sum],feed_dict={self.real_input: seq_train,self.label: label_train})
                    cls_accuracy_valid,cls_loss_value_valid=self.sess.run([self._cls_accuracy_op,self.cls_loss],feed_dict={self.real_input: seq_valid,self.label: label_valid})
                    if cls_loss_value_valid<0.8: 
                        self.cls_learning_rate = 1e-5
                        print('cls learning rate truned down 0.1fold') 
                    else:
                        self.cls_learning_rate = 1e-4
                    print("Epoch[%2d][%5d/%5d]:cls_loss_train:[%.8f],cls_loss_valid:[%.8f],cls_accuracy_train:[%.8f],cls_accuracy_valid:[%.8f]"%(epoch,idx,self.iteration,cls_loss_value,cls_loss_value_valid,cls_accuracy_train,cls_accuracy_valid))
                    self.save(self.checkpoint_dir, counter)
                    self.writer.add_summary(summary_str5, counter)
                    cls_accuracy_valid_all.append(cls_accuracy_valid)
                    cls_accuracy_train_all.append(cls_accuracy_train)
            fake_kmer = [kmer_statistics(i, self.Generator_seq()[1]) for i in [4,6,8]]
            fake_js = [fake_kmer[i].js_with(true_kmer[i]) for i in range(3)] 
            val_js = [val_kmer[i].js_with(fake_kmer[i]) for i in range(3)]
            fake_kmer_univ = [kmer_statistics(i, gen_univ_seq) for i in [4,6,8]]
            fake_js_univ = [fake_kmer_univ[i].js_with(true_kmer_univ[i]) for i in range(3)] 
            val_js_univ = [val_kmer_univ[i].js_with(fake_kmer_univ[i]) for i in range(3)]
            print('Epoch [{}]: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(
                    epoch,
                    fake_js[0],
                    fake_js[1],
                    fake_js[2]))
            print('Valid Dataset: js_4mer: {}, js_6mer: {}, js_8mer: {}'.format(
                    val_js[0],
                    val_js[1],
                    val_js[2]))
            print('Epoch [{}]: js_4mer_univ: {}, js_6mer_univ: {}, js_8mer_univ: {}'.format(
                    epoch,
                    fake_js_univ[0],
                    fake_js_univ[1],
                    fake_js_univ[2]))
            print('Valid Dataset: js_4mer_univ: {}, js_6mer_univ: {}, js_8mer_univ: {}'.format(
                    val_js_univ[0],
                    val_js_univ[1],
                    val_js_univ[2]))
            train_6js.append(fake_js[1])
            val_6js.append(val_js[1])
            train_6js_univ.append(fake_js_univ[1])
            val_6js_univ.append(val_js_univ[1])
            if best_js > val_js[1]:
                best_js = val_js[1]
                conv = 0
            else:
                if cls_accuracy_valid>0.9:
                    conv += 1
                    if conv > 20:
                        break
        
            self.save(self.checkpoint_dir, counter)
            if np.mod(epoch,20) == 0:
                pdf = PdfPages(log_dir+'/Allinput_6mer_JS_Distance.pdf')
                plt.figure()
                plt.plot(np.arange(len(train_6js)),train_6js)
                plt.plot(np.arange(len(val_6js)),val_6js)
                plt.plot([0,len(train_6js)-1],[val_js_val_true[1],val_js_val_true[1]])
                plt.legend(['JS_train','JS_valid','JS_control'])
                plt.ylabel('JS Distance')
                plt.xlabel('epoch')
                pdf.savefig()
                pdf.close()
                with open(self.log_dir+'/AAE_generated_rep', 'wb') as f:
                    pickle.dump(self.artifical_rep, f) 
                with open(self.log_dir+'/AAE_generated_label', 'wb') as f:
                    pickle.dump(self.artifical_spe, f) 
                pdf = PdfPages(log_dir+'/Univ_6mer_JS_Distance.pdf')
                plt.figure()
                plt.plot(np.arange(len(train_6js_univ)),train_6js_univ)
                plt.plot(np.arange(len(val_6js_univ)),val_6js_univ)
                plt.plot([0,len(train_6js_univ)-1],[val_js_val_true_univ[1],val_js_val_true_univ[1]])
                plt.legend(['JS_train','JS_valid','JS_control'])
                plt.ylabel('JS Distance')
                plt.xlabel('epoch')
                pdf.savefig()
                pdf.close()
                pdf = PdfPages(log_dir+'/cls_loss_byepoch.pdf')
                plt.figure()
                plt.plot(np.arange(len(cls_accuracy_train_all)),cls_accuracy_train_all)
                plt.plot(np.arange(len(cls_accuracy_valid_all)),cls_accuracy_valid_all)
                plt.legend(['cls_train','cls_valid'])
                plt.ylabel('cls loss(cross entropy)')
                plt.xlabel('epoch')
                pdf.savefig()
                pdf.close()

        self.save(self.checkpoint_dir, counter)

        return

    def read_train_data(self,batch_size,nbin,supervise_file,batch_dict_name=['seq_train', 'label_train','seq_valid','label_valid']):
        data = PromoterData('train',nbin,supervise_file,data_dir='./',shuffle=True,batch_dict_name=batch_dict_name) 
        data.setup(epoch_val=0, batch_size=batch_size)
        return data

    def get_cls_loss(self):
        EC=tf.expand_dims(self.layers['cls_logits_EC'], 1)
        PA=tf.expand_dims(self.layers['cls_logits_PA'], 1)
        logits = tf.concat((EC,PA),axis=1) 
        labels=tf.cast(self.label,tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='cross_entropy')
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def get_cls_accuracy(self):
        labels = self.label 
        ECresult = tf.expand_dims(self.layers['y_EC'],1)
        PAresult = tf.expand_dims(self.layers['y_PA'],1)
        cls_predict = tf.concat((ECresult,PAresult),axis=-1)
        cls_predict = tf.dtypes.cast(cls_predict, tf.float32)
        labels = tf.dtypes.cast(labels, tf.float32) 
        num_correct = tf.cast(tf.equal(labels, cls_predict), tf.float32) 
        return tf.reduce_mean(num_correct)

    def get_label(self):
        labels=tf.cast(self.label,tf.int64)
        labels_onehot = tf.one_hot(labels,self.nbin)
        return labels_onehot

    def get_predict_label(self):
        predict_logits=[self.layers['one_hot_y_approx_EC'],self.layers['one_hot_y_approx_PA']]
        return predict_logits
    
    def plot_distritbution(self,labels_onehot,predict_logits,epoch):
        epoch_count = int(epoch/self.plotfreq)
        if epoch_count <10:
            width = 0.35
            ind = np.arange(3)
            plt.figure(figsize=(20, 20))
            pdf = PdfPages(self.log_dir+'/predicted_EC_PA.pdf')
            print('[*] Plotting predict exp bin of EC_PA..')
            spelist=['EC','PA']
            for i in range(len(spelist)):
                    for j in range(2):
                        ax = plt.subplot(10,6,epoch_count*6+(i*2+j+1))
                        plt.bar(ind,np.reshape(labels_onehot[j,i,:],(1,2))[0], color='red',width=width)
                        plt.bar(ind,np.reshape(predict_logits[i][j,:],(1,2))[0],  edgecolor='k',facecolor='white',width=width)
                        plt.xlabel(spelist[i]+'_epoch_'+str(epoch))
            pdf.savefig() 
            pdf.close()
            print('[*] Plotting epoch '+str(epoch)+' exp predict done')
        return

    def cal_hamming(self,promoter_oh,reconstruction_seq_oh): 
        promoter_oh=np.argmax(promoter_oh, axis=2)
        hamm_dist_mean=np.sum(promoter_oh!=reconstruction_seq_oh)/len(reconstruction_seq_oh)
        return hamm_dist_mean 

    def inf_train_gen(self):
        while True:
            np.random.shuffle(self.data)
            for i in range(0, len(self.data)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield self.data[i:i+self.BATCH_SIZE,:,:]
                
    def save(self, checkpoint_dir, step):
        with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
            for c in self.charmap:
                f.write(c+'\t')

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name +'_step_'+str(step)+'.model'), global_step=step)


    def load(self, checkpoint_dir = './generative_model', model_name='aae'):
        print(" [*] Reading checkpoints...")
        
        with open(checkpoint_dir+ '/' + model_name + 'charmap.txt','r') as f:
            self.invcharmap = str.split(f.read())
            self.charmap = {}
            i=0
            for c in self.invcharmap:
                self.charmap[c] = i
                i+=1
        self.sess.run(tf.initialize_all_variables())
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

    def load_model(self,checkpoint_dir,epoch=30,name='Encoder'):
        checkpoint_dir_EncoderNet = os.path.join(checkpoint_dir,str(name)+'_'+str(epoch).format(epoch))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_EncoderNet)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(Encoder_sess, os.path.join(checkpoint_dir_EncoderNet, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return Encoder_sess
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def get_z(self,rep_dir,name_list_3,name_list_7,gen_bs=None,savename="",gen_num=3000,exp_flag=False):
        rep_data_7=[]
        speflag_7=[]
        for k in range(len(name_list_7)):
            fname=name_list_7[k]
            print(fname)
            with open(fname+'.pickle','rb') as f: 
                    promoter=pickle.load(f)
            with open(fname+'_label.pickle','rb') as f: 
                    label_in=pickle.load(f)
            print(len(promoter))
            print(label_in[0])
            if gen_bs is None:
                if len(promoter)>=256: 
                    gen_bs=256
                elif len(promoter)>=32: 
                    gen_bs=32
                else:
                    gen_bs=4
            promoter = seq2oh(promoter,self.charmap)
            for j in range(int(len(promoter)/gen_bs)):
                input_oh= promoter[j*gen_bs:(j+1)*gen_bs,:,:]
                input_label= label_in[j*gen_bs:(j+1)*gen_bs]
                each_data_z = self.Encoder_z(input_oh,datatype='oh')
                each_data = self.Generate_rep(input_label,each_data_z,gen_bs)
                print(np.array(each_data).shape)
                rep_data_7.append(each_data)
                speflag_7.append([fname]*each_data.shape[0]) 
        with open(self.log_dir+'/AAE_generated_rep', 'rb') as f:
            self.artifical_rep=pickle.load(f)
        with open(self.log_dir+'/AAE_generated_label', 'rb') as f:
            self.artifical_spe=pickle.load(f)
        print(np.array(self.artifical_rep).shape) 
        print(np.array(rep_data_7).shape) 
        print(np.array(speflag_7).shape)

        rep_data_7.append(self.artifical_rep)
        speflag_7.append(self.artifical_spe)
        rep_data_7=np.concatenate(rep_data_7) 
        speflag_7=np.concatenate(speflag_7)
        print('latent data_7 shape:')
        print(rep_data_7.shape)
        rep_data_file_7='rep_data_EC_PA_7_'+savename+'.pickle' 
        speflag_file_7='speflag_EC_PA_7_'+savename+'.pickle'
        with open(rep_dir+rep_data_file_7, 'wb') as f:
                pickle.dump(rep_data_7, f) 
        with open(rep_dir+speflag_file_7, 'wb') as f:
                pickle.dump(speflag_7, f) 
        print('saved!')
        if exp_flag:
            rep_data_3=[]
            speflag_3=[]
            exp_value_3=[]
            for k in range(len(name_list_3)):
                fname=name_list_3[k]
                print(fname)
                with open(fname+'.pickle','rb') as f: 
                        promoter=pickle.load(f)
                with open(fname+'_label.pickle','rb') as f: 
                        label_in=pickle.load(f)
                with open(fname+'_exp.pickle','rb') as f: 
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
                    each_data_z = self.Encoder_z(input_oh,datatype='oh')
                    each_data = self.Generate_rep(input_label,each_data_z,gen_bs)
                    rep_data_3.append(each_data)
                    speflag_3.append([fname]*each_data.shape[0]) 
                    exp_value_3.append(exp_in[j*gen_bs:(j+1)*gen_bs])
            rep_data_3=np.concatenate(rep_data_3) 
            speflag_3=np.concatenate(speflag_3)
            exp_value_3=np.concatenate(exp_value_3)
            print('latent data_3 shape:')
            print(rep_data_3.shape)
            rep_data_file_3='rep_data_EC_PA_3.pickle'
            speflag_file_3='speflag_EC_PA_3.pickle'
            exp_value_file_3='expvalue_EC_PA_3.pickle'
            with open(rep_dir+rep_data_file_3, 'wb') as f:
                    pickle.dump(rep_data_3, f) 
            with open(rep_dir+speflag_file_3, 'wb') as f:
                    pickle.dump(speflag_3, f) 
            with open(rep_dir+exp_value_file_3, 'wb') as f:
                    pickle.dump(exp_value_3, f) 
            print('saved!')
            return rep_data_3,rep_data_7,speflag_3,speflag_7,exp_value_3
        else:
            return rep_data_7,speflag_7


