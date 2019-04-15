import tensorflow as tf
import numpy as np
import ops

w_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.1)

pool_params = {
    "pool_size": 2,
    "strides": 2,
    "padding": 'valid'
}

def QNetwork(s1, s2, out_dim, name, reuse, sensor=True):    
    with tf.variable_scope(name, reuse=reuse):   
        h_conv1 = ops.conv2d(s1, filters=64, kernel_size=5, strides=(1,1), padding='same', activation=None, name='h1')
        h_conv1 = ops.lrelu(h_conv1)
        h_conv1 = tf.layers.max_pooling2d(h_conv1, **pool_params)
        
        h_conv2 = ops.conv2d(h_conv1, filters=128, kernel_size=3, strides=(1,1), padding='same', activation=None, name='h2')
        h_conv2 = ops.lrelu(h_conv2)
        h_conv2 = tf.layers.max_pooling2d(h_conv2, **pool_params)

        h_conv3 = ops.conv2d(h_conv2, filters=256, kernel_size=3, strides=(1,1), padding='same', activation=None, name='h3')
        h_conv3 = ops.lrelu(h_conv3)
        h_conv3 = tf.layers.max_pooling2d(h_conv3, **pool_params)

        h_re4 = tf.reshape(h_conv3, [-1,8*8*256])
        h_fc4 = ops.fc(h_re4, 256, activation=ops.lrelu, name='h4')

        if sensor:
            h_s1 = ops.fc(s2, 32, activation=ops.lrelu, name='hs1')
            h_fc4 = tf.concat([h_fc4, h_s1], axis=1)

        out_logit = ops.fc(h_fc4, out_dim, activation=None, name='h5')
    return out_logit