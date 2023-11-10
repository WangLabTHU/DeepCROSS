import tensorflow as tf
import numpy as np
#from .linear import Linear

def scaled_dot_product_attention(Q, K, V, 
                                 causality=False, 
                                 dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  
        outputs /= d_k ** 0.5
        outputs = tf.nn.softmax(outputs) 
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)  
    return outputs


def multihead_attention(queries, keys, values, 
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    d_model = queries.get_shape().as_list()[-1] 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=True) 
        K = tf.layers.dense(keys, d_model, use_bias=True) 
        V = tf.layers.dense(values, d_model, use_bias=True) 
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) 
        outputs = scaled_dot_product_attention(Q_, K_, V_,causality, dropout_rate, training)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) 
        outputs += queries
        outputs = ln(outputs)
    return outputs 


def ff(inputs, num_units=[2048,512], scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs
        outputs = ln(outputs)
    return outputs


def ln(inputs, epsilon = 1e-8, scope="ln"): 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
    return outputs

def positional_encoding(inputs,
                        maxlen=165, 
                        masking=True,
                        scope="positional_encoding"): 
    E = inputs.get_shape().as_list()[-1] 
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) 
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) 
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        return tf.to_float(outputs)

