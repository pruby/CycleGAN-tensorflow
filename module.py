from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def centre_crop(image, width, height):
    offset_x = (int(image.shape[1]) - int(width)) // 2
    offset_y = (int(image.shape[2]) - int(height)) // 2
    return tf.image.crop_to_bounding_box(image, offset_x, offset_y, int(width), int(height))

def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def generator_unet(image, options, reuse=False, name="generator"):
    print("Generator %s" % (name,))
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        layers = []
        last = image
        dim = options.df_dim
        for i in range(4):
            last = instance_norm(last, 'g_bn%d_d' % (i,))
            p1 = tf.layers.conv2d(last, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name=('g_l%d_p1_conv' % (i,)))
            print("Generator layer: " + str(p1.shape))
            p2 = tf.layers.conv2d(p1, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name=('g_l%d_p2_conv' % (i,)))
            print("Generator layer: " + str(p2.shape))
            layers.append(p2)
            downscale = tf.layers.max_pooling2d(p2, (2, 2), 2, name=('g_l%d_downscale' % (i,)))
            print("Generator layer: " + str(downscale.shape))
            last = downscale
            if options.is_training:
                last = tf.layers.dropout(last, rate=0.5)
            if i < 3:
              dim = dim * 2

        b1 = tf.layers.conv2d(last, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name='g_b1')
        print("Generator layer: " + str(b1.shape))
        b2 = tf.layers.conv2d(b1, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name='g_b2')
        print("Generator layer: " + str(b2.shape))
        last = b2

        layers.reverse()
        i = len(layers)
        for layer in layers:
            last = instance_norm(last, 'g_bn%d_u' % (i,))
            i -= 1 
            upscale = tf.layers.conv2d_transpose(last, dim, (2, 2), 2, kernel_initializer=initializer, activation=tf.nn.leaky_relu, name=("g_l%d_up" % (i,)))
            width = int(upscale.shape[1])
            height = int(upscale.shape[2])
            dim = layer.shape[3]
            center = centre_crop(layer, width, height)
            combined = tf.concat([upscale, center], -1)
            print("Generator layer: " + str(combined.shape))
            p3 = tf.layers.conv2d(combined, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name=('g_l%d_p3_conv' % (i,)))
            print("Generator layer: " + str(p3.shape))
            p4 = tf.layers.conv2d(p3, dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.leaky_relu, name=('g_l%d_p4_conv' % (i,)))
            print("Generator layer: " + str(p4.shape))
            last = p4
            if options.is_training:
                last = tf.layers.dropout(last, rate=0.5)

        final = tf.layers.conv2d(last, options.output_c_dim, (3, 3), kernel_initializer=initializer, activation=tf.nn.tanh, name="g_final_projection")

        return final


def abs_criterion(in_, target):
    # Compare in_ to the equivalent area in target
    in_ = centre_crop(in_, target.shape[1], target.shape[2])
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
