import tensorflow as tf
import numpy as np

w_init = tf.contrib.layers.xavier_initializer()
b_init = tf.constant_initializer(0.1)

def bideconv2d(x, filters, kernel_size, strides, padding, activation, name):
    shape_list = x.get_shape()
    x_up = tf.image.resize_images(x, [shape_list[1]*strides[0], shape_list[2]*strides[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.layers.conv2d(
            x_up, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=(1,1), 
            padding=padding, 
            activation=activation, 
            kernel_initializer=w_init, 
            bias_initializer=b_init, 
            name=name)

def conv2d(x, filters, kernel_size, strides, padding, activation, name):
    return tf.layers.conv2d(
            x, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation,
            kernel_initializer=w_init, 
            bias_initializer=b_init, 
            name=name)

def bn(x, training, name):
    return tf.contrib.layers.batch_norm(
            x,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=training,
            scope=name
    )

def norm(x, norm_type, training, name='norm', G=16, esp=1e-5):
    with tf.variable_scope('{}_{}_norm'.format(name,norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.9,
                is_training=training, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            print(name, x.get_shape().as_list(), x.shape)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output

def fc(x, units, activation, name):
    return tf.layers.dense(x, units, activation=activation, kernel_initializer=w_init, bias_initializer=b_init, name=name)

def lrelu(x, leak=0.2):
	return tf.maximum(x, leak*x)