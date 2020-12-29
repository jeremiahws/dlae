

import tensorflow as tf
import numpy as np
import os
import argparse
import h5py
from src.utils.data_generators import FCN2DDatasetGenerator


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True, scope=self.name)


def conv2d(image, output_dim, k_size=5, stride=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        if name[0:2] == 'g_':
            w = tf.get_variable("kernel", shape=[k_size, k_size, image.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                regularizer=orthogonal_regularizer(0.0001))
        else:
            w = tf.get_variable("kernel", shape=[k_size, k_size, image.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                regularizer=None)

        x = tf.nn.conv2d(input=image, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME')
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

        return x


def tpconv2d(image, output_shape, k_size=5, stride=2, stddev=0.02, name='tpconv2d', with_w=False):
    with tf.variable_scope(name):
        x_shape = image.get_shape().as_list()

        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, output_shape[-1]]

        w = tf.get_variable("kernel", shape=[k_size, k_size, output_shape[-1], image.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                            regularizer=orthogonal_regularizer(0.0001))
        x = tf.nn.conv2d_transpose(image, filter=spectral_norm(w),
                                   output_shape=output_shape, strides=[1, stride, stride, 1])

        bias = tf.get_variable("bias", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

        if with_w:
            return x, w, bias
        else:
            return x


def orthogonal_regularizer(scale):
    def ortho_reg(w) :
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])
        identity = tf.eye(c)

        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def linear(tensor, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = tensor.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(tensor, matrix) + bias, matrix, bias
        else:
           return tf.matmul(tensor, matrix) + bias


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


class CascadedCGAN(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, output_size=256,
                 gf_dim=64, df_dim=64, l1_lambda=100,
                 input_c_dim=60, output_c_dim=1,
                 checkpoint_dir=None,
                 load_checkpoint=False,
                 train_data_gen=None,
                 valid_data_gen=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.l1_lambda = l1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        self.g_bn_d8 = batch_norm(name='g_bn_d8')

        self.g_bn_e1_2 = batch_norm(name='g_bn_e1_2')
        self.g_bn_e2_2 = batch_norm(name='g_bn_e2_2')
        self.g_bn_e3_2 = batch_norm(name='g_bn_e3_2')
        self.g_bn_e4_2 = batch_norm(name='g_bn_e4_2')
        self.g_bn_e5_2 = batch_norm(name='g_bn_e5_2')
        self.g_bn_e6_2 = batch_norm(name='g_bn_e6_2')
        self.g_bn_e7_2 = batch_norm(name='g_bn_e7_2')
        self.g_bn_e8_2 = batch_norm(name='g_bn_e8_2')

        self.g_bn_d1_2 = batch_norm(name='g_bn_d1_2')
        self.g_bn_d2_2 = batch_norm(name='g_bn_d2_2')
        self.g_bn_d3_2 = batch_norm(name='g_bn_d3_2')
        self.g_bn_d4_2 = batch_norm(name='g_bn_d4_2')
        self.g_bn_d5_2 = batch_norm(name='g_bn_d5_2')
        self.g_bn_d6_2 = batch_norm(name='g_bn_d6_2')
        self.g_bn_d7_2 = batch_norm(name='g_bn_d7_2')
        self.g_bn_d8_2 = batch_norm(name='g_bn_d8_2')

        self.g_bn_e1_3 = batch_norm(name='g_bn_e1_3')
        self.g_bn_e2_3 = batch_norm(name='g_bn_e2_3')
        self.g_bn_e3_3 = batch_norm(name='g_bn_e3_3')
        self.g_bn_e4_3 = batch_norm(name='g_bn_e4_3')
        self.g_bn_e5_3 = batch_norm(name='g_bn_e5_3')
        self.g_bn_e6_3 = batch_norm(name='g_bn_e6_3')
        self.g_bn_e7_3 = batch_norm(name='g_bn_e7_3')
        self.g_bn_e8_3 = batch_norm(name='g_bn_e8_3')

        self.g_bn_d1_3 = batch_norm(name='g_bn_d1_3')
        self.g_bn_d2_3 = batch_norm(name='g_bn_d2_3')
        self.g_bn_d3_3 = batch_norm(name='g_bn_d3_3')
        self.g_bn_d4_3 = batch_norm(name='g_bn_d4_3')
        self.g_bn_d5_3 = batch_norm(name='g_bn_d5_3')
        self.g_bn_d6_3 = batch_norm(name='g_bn_d6_3')
        self.g_bn_d7_3 = batch_norm(name='g_bn_d7_3')

        self.checkpoint_dir = checkpoint_dir
        self.load_checkpoint = load_checkpoint

        self.train_batches = len(train_data_gen)
        self.train_data_gen = train_data_gen.generate()
        self.valid_data_gen = valid_data_gen.generate()

        self.build_model()

    def build_model(self):
        self.train_data = tf.placeholder(tf.float32,
                                         [self.batch_size, self.image_size, self.image_size,
                                          self.input_c_dim + self.output_c_dim],
                                         name='real_dce_and_bv_images_train')

        self.val_data = tf.placeholder(tf.float32,
                                       [self.batch_size, self.image_size, self.image_size,
                                        self.input_c_dim + self.output_c_dim],
                                       name='real_dce_and_bv_images_val')

        self.real_dce_t = self.train_data[:, :, :, :self.input_c_dim]
        self.real_bv_t = self.train_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.real_dce_v = self.val_data[:, :, :, :self.input_c_dim]
        self.real_bv_v = self.val_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_bv_t = self.generator(self.real_dce_t)

        self.real_dceANDbv = tf.concat([self.real_dce_t, self.real_bv_t], 3)
        self.fake_dceANDbv = tf.concat([self.real_dce_t, self.fake_bv_t], 3)
        self.D, self.D_logits = self.discriminator(self.real_dceANDbv, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_dceANDbv, reuse=True)

        self.fake_bv_t_sample = self.sampler(self.real_dce_t)
        self.fake_bv_v_sample = self.sampler(self.real_dce_v)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                                  labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                                  labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.l1_penalty = self.l1_lambda * tf.reduce_mean(tf.abs(self.real_bv_t - self.fake_bv_t))
        self.l1_penalty_v = self.l1_lambda * tf.reduce_mean(tf.abs(self.real_bv_v - self.fake_bv_v_sample))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                             labels=tf.ones_like(self.D_))) + self.l1_penalty

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.bv_t_sum = tf.summary.image('real_vs_fake_bv_train', tf.concat([self.real_bv_t, self.fake_bv_t_sample], 2))
        self.dce_t_ex = tf.concat([self.real_dce_t[:, :, :, 5],
                                   self.real_dce_t[:, :, :, 10],
                                   self.real_dce_t[:, :, :, 25],
                                   self.real_dce_t[:, :, :, 40]], 2)
        self.dce_t_ex = tf.expand_dims(self.dce_t_ex, axis=-1)
        self.dce_t_sum = tf.summary.image('dce_input_train', self.dce_t_ex)

        self.bv_v_sum = tf.summary.image('real_vs_fake_bv_val', tf.concat([self.real_bv_v, self.fake_bv_v_sample], 2))
        self.dce_v_ex = tf.concat([self.real_dce_v[:, :, :, 5],
                                   self.real_dce_v[:, :, :, 10],
                                   self.real_dce_v[:, :, :, 25],
                                   self.real_dce_v[:, :, :, 40]], 2)
        self.dce_v_ex = tf.expand_dims(self.dce_v_ex, axis=-1)
        self.dce_v_sum = tf.summary.image('dce_input_val', self.dce_v_ex)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.l1_penalty_sum = tf.summary.scalar("l1_penalty", self.l1_penalty)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.l1_penalty_sum_v = tf.summary.scalar("l1_penalty_v", self.l1_penalty_v)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train_graph(self, lr=0.0002, beta1=0.5, epochs=100):
        d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.bv_t_sum,
                                       self.dce_t_sum, self.bv_v_sum,
                                       self.dce_v_sum, self.d_loss_fake_sum,
                                       self.g_loss_sum, self.l1_penalty_sum,
                                       self.l1_penalty_sum_v])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1

        if self.load_checkpoint is True:
            self.load_model(self.checkpoint_dir)

        for epoch in range(epochs):
            for idx in range(self.train_batches):
                t_data = next(self.train_data_gen)
                train_sample = np.concatenate((t_data[0], t_data[1]), axis=-1)

                v_data = next(self.valid_data_gen)
                valid_sample = np.concatenate((v_data[0], v_data[1]), axis=-1)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.train_data: train_sample})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.train_data: train_sample,
                                                                                 self.val_data: valid_sample})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.train_data: train_sample,
                                                                                 self.val_data: valid_sample})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.train_data: train_sample})
                errD_real = self.d_loss_real.eval({self.train_data: train_sample})
                errG = self.g_loss.eval({self.train_data: train_sample})

                print(errD_fake, errD_real, errG)

                counter += 1
                #TODO print summary

                if np.mod(counter, 500) == 2:
                    self.save_model(self.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = leaky_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = leaky_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = leaky_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = leaky_relu(self.d_bn3(conv2d(h2, self.df_dim*8, stride=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image):
        with tf.variable_scope("generator") as scope:
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(leaky_relu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(leaky_relu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(leaky_relu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(leaky_relu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(leaky_relu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(leaky_relu(e6), self.gf_dim*8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(leaky_relu(e7), self.gf_dim*8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = tpconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim*8],
                                                     name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 = tf.concat([self.g_bn_d1(self.d1), e7], 3)

            self.d2, self.d2_w, self.d2_b = tpconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8],
                                                     name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 = tf.concat([self.g_bn_d2(self.d2), e6], 3)

            self.d3, self.d3_w, self.d3_b = tpconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8],
                                                     name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 = tf.concat([self.g_bn_d3(self.d3), e5], 3)

            self.d4, self.d4_w, self.d4_b = tpconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim*8],
                                                     name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = tpconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim*4],
                                                     name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = tpconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim*2],
                                                     name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = tpconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim],
                                                     name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = tpconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim],
                                                     name='g_d8', with_w=True)
            d8 = self.g_bn_d8(self.d8)

            e1_2 = self.g_bn_e1_2(conv2d(leaky_relu(d8), self.gf_dim, name='g_e1_conv_2'))
            e2_2 = self.g_bn_e2_2(conv2d(leaky_relu(e1_2), self.gf_dim * 2, name='g_e2_conv_2'))
            e3_2 = self.g_bn_e3_2(conv2d(leaky_relu(e2_2), self.gf_dim * 4, name='g_e3_conv_2'))
            e4_2 = self.g_bn_e4_2(conv2d(leaky_relu(e3_2), self.gf_dim * 8, name='g_e4_conv_2'))
            e5_2 = self.g_bn_e5_2(conv2d(leaky_relu(e4_2), self.gf_dim * 8, name='g_e5_conv_2'))
            e6_2 = self.g_bn_e6_2(conv2d(leaky_relu(e5_2), self.gf_dim * 8, name='g_e6_conv_2'))
            e7_2 = self.g_bn_e7_2(conv2d(leaky_relu(e6_2), self.gf_dim * 8, name='g_e7_conv_2'))
            e8_2 = self.g_bn_e8_2(conv2d(leaky_relu(e7_2), self.gf_dim * 8, name='g_e8_conv_2'))

            self.d1_2, self.d1_w_2, self.d1_b_2 = tpconv2d(tf.nn.relu(e8_2),
                                                           [self.batch_size, s128, s128, self.gf_dim * 8],
                                                           name='g_d1_2', with_w=True)
            d1_2 = tf.nn.dropout(self.g_bn_d1_2(self.d1_2), 0.5)
            d1_2 = tf.concat([d1_2, e7_2], 3)
            # d1_2 = tf.concat([self.g_bn_d1_2(self.d1_2), e7_2], 3)

            self.d2_2, self.d2_w_2, self.d2_b_2 = tpconv2d(tf.nn.relu(d1_2),
                                                           [self.batch_size, s64, s64, self.gf_dim * 8],
                                                           name='g_d2_2', with_w=True)
            d2_2 = tf.nn.dropout(self.g_bn_d2_2(self.d2_2), 0.5)
            d2_2 = tf.concat([d2_2, e6_2], 3)
            # d2_2 = tf.concat([self.g_bn_d2_2(self.d2_2), e6_2], 3)

            self.d3_2, self.d3_w_2, self.d3_b_2 = tpconv2d(tf.nn.relu(d2_2),
                                                           [self.batch_size, s32, s32, self.gf_dim * 8],
                                                           name='g_d3_2', with_w=True)
            d3_2 = tf.nn.dropout(self.g_bn_d3_2(self.d3_2), 0.5)
            d3_2 = tf.concat([d3_2, e5_2], 3)
            # d3_2 = tf.concat([self.g_bn_d3_2(self.d3_2), e5_2], 3)

            self.d4_2, self.d4_w_2, self.d4_b_2 = tpconv2d(tf.nn.relu(d3_2),
                                                           [self.batch_size, s16, s16, self.gf_dim * 8],
                                                           name='g_d4_2', with_w=True)
            d4_2 = self.g_bn_d4_2(self.d4_2)
            d4_2 = tf.concat([d4_2, e4_2], 3)

            self.d5_2, self.d5_w_2, self.d5_b_2 = tpconv2d(tf.nn.relu(d4_2),
                                                           [self.batch_size, s8, s8, self.gf_dim * 4],
                                                           name='g_d5_2', with_w=True)
            d5_2 = self.g_bn_d5_2(self.d5_2)
            d5_2 = tf.concat([d5_2, e3_2], 3)

            self.d6_2, self.d6_w_2, self.d6_b_2 = tpconv2d(tf.nn.relu(d5_2),
                                                           [self.batch_size, s4, s4, self.gf_dim * 2],
                                                           name='g_d6_2', with_w=True)
            d6_2 = self.g_bn_d6_2(self.d6_2)
            d6_2 = tf.concat([d6_2, e2_2], 3)

            self.d7_2, self.d7_w_2, self.d7_b_2 = tpconv2d(tf.nn.relu(d6_2),
                                                           [self.batch_size, s2, s2, self.gf_dim],
                                                           name='g_d7_2', with_w=True)
            d7_2 = self.g_bn_d7_2(self.d7_2)
            d7_2 = tf.concat([d7_2, e1_2], 3)

            self.d8_2, self.d8_w_2, self.d8_b_2 = tpconv2d(tf.nn.relu(d7_2),
                                                           [self.batch_size, s, s, self.output_c_dim],
                                                           name='g_d8_2', with_w=True)
            # d8_2 = self.g_bn_d8_2(self.d8_2)
            #
            # e1_3 = self.g_bn_e1_3(conv2d(leaky_relu(d8_2), self.gf_dim, name='g_e1_conv_3'))
            # e2_3 = self.g_bn_e2_3(conv2d(leaky_relu(e1_3), self.gf_dim * 2, name='g_e2_conv_3'))
            # e3_3 = self.g_bn_e3_3(conv2d(leaky_relu(e2_3), self.gf_dim * 4, name='g_e3_conv_3'))
            # e4_3 = self.g_bn_e4_3(conv2d(leaky_relu(e3_3), self.gf_dim * 8, name='g_e4_conv_3'))
            # e5_3 = self.g_bn_e5_3(conv2d(leaky_relu(e4_3), self.gf_dim * 8, name='g_e5_conv_3'))
            # e6_3 = self.g_bn_e6_3(conv2d(leaky_relu(e5_3), self.gf_dim * 8, name='g_e6_conv_3'))
            # e7_3 = self.g_bn_e7_3(conv2d(leaky_relu(e6_3), self.gf_dim * 8, name='g_e7_conv_3'))
            # e8_3 = self.g_bn_e8_3(conv2d(leaky_relu(e7_3), self.gf_dim * 8, name='g_e8_conv_3'))
            #
            # self.d1_3, self.d1_w_3, self.d1_b_3 = tpconv2d(tf.nn.relu(e8_3),
            #                                                [self.batch_size, s128, s128, self.gf_dim * 8],
            #                                                name='g_d1_3', with_w=True)
            # d1_3 = tf.nn.dropout(self.g_bn_d1_3(self.d1_3), 0.5)
            # d1_3 = tf.concat([d1_3, e7_3], 3)
            #
            # self.d2_3, self.d2_w_3, self.d2_b_3 = tpconv2d(tf.nn.relu(d1_3),
            #                                                [self.batch_size, s64, s64, self.gf_dim * 8],
            #                                                name='g_d2_3', with_w=True)
            # d2_3 = tf.nn.dropout(self.g_bn_d2_3(self.d2_3), 0.5)
            # d2_3 = tf.concat([d2_3, e6_3], 3)
            #
            # self.d3_3, self.d3_w_3, self.d3_b_3 = tpconv2d(tf.nn.relu(d2_3),
            #                                                [self.batch_size, s32, s32, self.gf_dim * 8],
            #                                                name='g_d3_3', with_w=True)
            # d3_3 = tf.nn.dropout(self.g_bn_d3_3(self.d3_3), 0.5)
            # d3_3 = tf.concat([d3_3, e5_3], 3)
            #
            # self.d4_3, self.d4_w_3, self.d4_b_3 = tpconv2d(tf.nn.relu(d3_3),
            #                                                [self.batch_size, s16, s16, self.gf_dim * 8],
            #                                                name='g_d4_3', with_w=True)
            # d4_3 = self.g_bn_d4_3(self.d4_3)
            # d4_3 = tf.concat([d4_3, e4_3], 3)
            #
            # self.d5_3, self.d5_w_3, self.d5_b_3 = tpconv2d(tf.nn.relu(d4_3),
            #                                                [self.batch_size, s8, s8, self.gf_dim * 4],
            #                                                name='g_d5_3', with_w=True)
            # d5_3 = self.g_bn_d5_3(self.d5_3)
            # d5_3 = tf.concat([d5_3, e3_3], 3)
            #
            # self.d6_3, self.d6_w_3, self.d6_b_3 = tpconv2d(tf.nn.relu(d5_3),
            #                                                [self.batch_size, s4, s4, self.gf_dim * 2],
            #                                                name='g_d6_3', with_w=True)
            # d6_3 = self.g_bn_d6_3(self.d6_3)
            # d6_3 = tf.concat([d6_3, e2_3], 3)
            #
            # self.d7_3, self.d7_w_3, self.d7_b_3 = tpconv2d(tf.nn.relu(d6_3),
            #                                                [self.batch_size, s2, s2, self.gf_dim],
            #                                                name='g_d7_3', with_w=True)
            # d7_3 = self.g_bn_d7_3(self.d7_3)
            # d7_3 = tf.concat([d7_3, e1_3], 3)
            #
            # self.d8_3, self.d8_w_3, self.d8_b_3 = tpconv2d(tf.nn.relu(d7_3),
            #                                                [self.batch_size, s, s, self.output_c_dim],
            #                                                name='g_d8_3', with_w=True)
            #
            # return tf.nn.tanh(self.d8_3)
            return tf.nn.tanh(self.d8_2)

    def sampler(self, image):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(leaky_relu(e1), self.gf_dim * 2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(leaky_relu(e2), self.gf_dim * 4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(leaky_relu(e3), self.gf_dim * 8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(leaky_relu(e4), self.gf_dim * 8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(leaky_relu(e5), self.gf_dim * 8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(leaky_relu(e6), self.gf_dim * 8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(leaky_relu(e7), self.gf_dim * 8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = tpconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim * 8],
                                                     name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 = tf.concat([self.g_bn_d1(self.d1), e7], 3)

            self.d2, self.d2_w, self.d2_b = tpconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8],
                                                     name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 = tf.concat([self.g_bn_d2(self.d2), e6], 3)

            self.d3, self.d3_w, self.d3_b = tpconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8],
                                                     name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 = tf.concat([self.g_bn_d3(self.d3), e5], 3)

            self.d4, self.d4_w, self.d4_b = tpconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim * 8],
                                                     name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = tpconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim * 4],
                                                     name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = tpconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim * 2],
                                                     name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = tpconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim],
                                                     name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = tpconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim],
                                                     name='g_d8', with_w=True)
            d8 = self.g_bn_d8(self.d8)

            e1_2 = self.g_bn_e1_2(conv2d(leaky_relu(d8), self.gf_dim, name='g_e1_conv_2'))
            e2_2 = self.g_bn_e2_2(conv2d(leaky_relu(e1_2), self.gf_dim * 2, name='g_e2_conv_2'))
            e3_2 = self.g_bn_e3_2(conv2d(leaky_relu(e2_2), self.gf_dim * 4, name='g_e3_conv_2'))
            e4_2 = self.g_bn_e4_2(conv2d(leaky_relu(e3_2), self.gf_dim * 8, name='g_e4_conv_2'))
            e5_2 = self.g_bn_e5_2(conv2d(leaky_relu(e4_2), self.gf_dim * 8, name='g_e5_conv_2'))
            e6_2 = self.g_bn_e6_2(conv2d(leaky_relu(e5_2), self.gf_dim * 8, name='g_e6_conv_2'))
            e7_2 = self.g_bn_e7_2(conv2d(leaky_relu(e6_2), self.gf_dim * 8, name='g_e7_conv_2'))
            e8_2 = self.g_bn_e8_2(conv2d(leaky_relu(e7_2), self.gf_dim * 8, name='g_e8_conv_2'))

            self.d1_2, self.d1_w_2, self.d1_b_2 = tpconv2d(tf.nn.relu(e8_2),
                                                           [self.batch_size, s128, s128, self.gf_dim * 8],
                                                           name='g_d1_2', with_w=True)
            d1_2 = tf.nn.dropout(self.g_bn_d1_2(self.d1_2), 0.5)
            d1_2 = tf.concat([d1_2, e7_2], 3)
            # d1_2 = tf.concat([self.g_bn_d1_2(self.d1_2), e7_2], 3)

            self.d2_2, self.d2_w_2, self.d2_b_2 = tpconv2d(tf.nn.relu(d1_2),
                                                           [self.batch_size, s64, s64, self.gf_dim * 8],
                                                           name='g_d2_2', with_w=True)
            d2_2 = tf.nn.dropout(self.g_bn_d2_2(self.d2_2), 0.5)
            d2_2 = tf.concat([d2_2, e6_2], 3)
            # d2_2 = tf.concat([self.g_bn_d2_2(self.d2_2), e6_2], 3)

            self.d3_2, self.d3_w_2, self.d3_b_2 = tpconv2d(tf.nn.relu(d2_2),
                                                           [self.batch_size, s32, s32, self.gf_dim * 8],
                                                           name='g_d3_2', with_w=True)
            d3_2 = tf.nn.dropout(self.g_bn_d3_2(self.d3_2), 0.5)
            d3_2 = tf.concat([d3_2, e5_2], 3)
            # d3_2 = tf.concat([self.g_bn_d3_2(self.d3_2), e5_2], 3)

            self.d4_2, self.d4_w_2, self.d4_b_2 = tpconv2d(tf.nn.relu(d3_2),
                                                           [self.batch_size, s16, s16, self.gf_dim * 8],
                                                           name='g_d4_2', with_w=True)
            d4_2 = self.g_bn_d4_2(self.d4_2)
            d4_2 = tf.concat([d4_2, e4_2], 3)

            self.d5_2, self.d5_w_2, self.d5_b_2 = tpconv2d(tf.nn.relu(d4_2),
                                                           [self.batch_size, s8, s8, self.gf_dim * 4],
                                                           name='g_d5_2', with_w=True)
            d5_2 = self.g_bn_d5_2(self.d5_2)
            d5_2 = tf.concat([d5_2, e3_2], 3)

            self.d6_2, self.d6_w_2, self.d6_b_2 = tpconv2d(tf.nn.relu(d5_2),
                                                           [self.batch_size, s4, s4, self.gf_dim * 2],
                                                           name='g_d6_2', with_w=True)
            d6_2 = self.g_bn_d6_2(self.d6_2)
            d6_2 = tf.concat([d6_2, e2_2], 3)

            self.d7_2, self.d7_w_2, self.d7_b_2 = tpconv2d(tf.nn.relu(d6_2),
                                                           [self.batch_size, s2, s2, self.gf_dim],
                                                           name='g_d7_2', with_w=True)
            d7_2 = self.g_bn_d7_2(self.d7_2)
            d7_2 = tf.concat([d7_2, e1_2], 3)

            self.d8_2, self.d8_w_2, self.d8_b_2 = tpconv2d(tf.nn.relu(d7_2),
                                                           [self.batch_size, s, s, self.output_c_dim],
                                                           name='g_d8_2', with_w=True)
            # d8_2 = self.g_bn_d8_2(self.d8_2)
            #
            # e1_3 = self.g_bn_e1_3(conv2d(leaky_relu(d8_2), self.gf_dim, name='g_e1_conv_3'))
            # e2_3 = self.g_bn_e2_3(conv2d(leaky_relu(e1_3), self.gf_dim * 2, name='g_e2_conv_3'))
            # e3_3 = self.g_bn_e3_3(conv2d(leaky_relu(e2_3), self.gf_dim * 4, name='g_e3_conv_3'))
            # e4_3 = self.g_bn_e4_3(conv2d(leaky_relu(e3_3), self.gf_dim * 8, name='g_e4_conv_3'))
            # e5_3 = self.g_bn_e5_3(conv2d(leaky_relu(e4_3), self.gf_dim * 8, name='g_e5_conv_3'))
            # e6_3 = self.g_bn_e6_3(conv2d(leaky_relu(e5_3), self.gf_dim * 8, name='g_e6_conv_3'))
            # e7_3 = self.g_bn_e7_3(conv2d(leaky_relu(e6_3), self.gf_dim * 8, name='g_e7_conv_3'))
            # e8_3 = self.g_bn_e8_3(conv2d(leaky_relu(e7_3), self.gf_dim * 8, name='g_e8_conv_3'))
            #
            # self.d1_3, self.d1_w_3, self.d1_b_3 = tpconv2d(tf.nn.relu(e8_3),
            #                                                [self.batch_size, s128, s128, self.gf_dim * 8],
            #                                                name='g_d1_3', with_w=True)
            # d1_3 = tf.nn.dropout(self.g_bn_d1_3(self.d1_3), 0.5)
            # d1_3 = tf.concat([d1_3, e7_3], 3)
            #
            # self.d2_3, self.d2_w_3, self.d2_b_3 = tpconv2d(tf.nn.relu(d1_3),
            #                                                [self.batch_size, s64, s64, self.gf_dim * 8],
            #                                                name='g_d2_3', with_w=True)
            # d2_3 = tf.nn.dropout(self.g_bn_d2_3(self.d2_3), 0.5)
            # d2_3 = tf.concat([d2_3, e6_3], 3)
            #
            # self.d3_3, self.d3_w_3, self.d3_b_3 = tpconv2d(tf.nn.relu(d2_3),
            #                                                [self.batch_size, s32, s32, self.gf_dim * 8],
            #                                                name='g_d3_3', with_w=True)
            # d3_3 = tf.nn.dropout(self.g_bn_d3_3(self.d3_3), 0.5)
            # d3_3 = tf.concat([d3_3, e5_3], 3)
            #
            # self.d4_3, self.d4_w_3, self.d4_b_3 = tpconv2d(tf.nn.relu(d3_3),
            #                                                [self.batch_size, s16, s16, self.gf_dim * 8],
            #                                                name='g_d4_3', with_w=True)
            # d4_3 = self.g_bn_d4_3(self.d4_3)
            # d4_3 = tf.concat([d4_3, e4_3], 3)
            #
            # self.d5_3, self.d5_w_3, self.d5_b_3 = tpconv2d(tf.nn.relu(d4_3),
            #                                                [self.batch_size, s8, s8, self.gf_dim * 4],
            #                                                name='g_d5_3', with_w=True)
            # d5_3 = self.g_bn_d5_3(self.d5_3)
            # d5_3 = tf.concat([d5_3, e3_3], 3)
            #
            # self.d6_3, self.d6_w_3, self.d6_b_3 = tpconv2d(tf.nn.relu(d5_3),
            #                                                [self.batch_size, s4, s4, self.gf_dim * 2],
            #                                                name='g_d6_3', with_w=True)
            # d6_3 = self.g_bn_d6_3(self.d6_3)
            # d6_3 = tf.concat([d6_3, e2_3], 3)
            #
            # self.d7_3, self.d7_w_3, self.d7_b_3 = tpconv2d(tf.nn.relu(d6_3),
            #                                                [self.batch_size, s2, s2, self.gf_dim],
            #                                                name='g_d7_3', with_w=True)
            # d7_3 = self.g_bn_d7_3(self.d7_3)
            # d7_3 = tf.concat([d7_3, e1_3], 3)
            #
            # self.d8_3, self.d8_w_3, self.d8_b_3 = tpconv2d(tf.nn.relu(d7_3),
            #                                                [self.batch_size, s, s, self.output_c_dim],
            #                                                name='g_d8_3', with_w=True)
            #
            # return tf.nn.tanh(self.d8_3)
            return tf.nn.tanh(self.d8_2)

    def save_model(self, checkpoint_dir, step):
        model_name = "cGAN.model"
        model_dir = "%s_%s_%s" % ('test', self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_model(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

    def predict_on_graph(self, test_gen=None, file_name=None):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.load_model(self.checkpoint_dir)

        n_samples = len(test_gen)
        data = test_gen.generate()

        f = h5py.File(file_name, 'w')
        for i in range(n_samples):
            sample = next(data)
            sample_images = sample[0]
            sample_images = sample_images[np.newaxis, :, :, :]
            blank_bv = np.zeros((self.batch_size, self.image_size, self.image_size, self.output_c_dim))
            sample_images = np.concatenate((sample_images, blank_bv), axis=-1)
            samples = self.sess.run(self.fake_bv_t_sample, feed_dict={self.train_data: sample_images})
            f.create_dataset(str(i), data=samples)

        f.close()


class CGAN(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, output_size=256,
                 gf_dim=64, df_dim=64, l1_lambda=100,
                 input_c_dim=60, output_c_dim=1,
                 checkpoint_dir=None,
                 load_checkpoint=False,
                 train_data_gen=None,
                 valid_data_gen=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.l1_lambda = l1_lambda

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.checkpoint_dir = checkpoint_dir
        self.load_checkpoint = load_checkpoint

        self.train_batches = len(train_data_gen)
        self.train_data_gen = train_data_gen.generate()
        self.valid_data_gen = valid_data_gen.generate()

        self.build_model()

    def build_model(self):
        self.train_data = tf.placeholder(tf.float32,
                                         [self.batch_size, self.image_size, self.image_size,
                                          self.input_c_dim + self.output_c_dim],
                                         name='real_dce_and_bv_images_train')

        self.val_data = tf.placeholder(tf.float32,
                                       [self.batch_size, self.image_size, self.image_size,
                                        self.input_c_dim + self.output_c_dim],
                                       name='real_dce_and_bv_images_val')

        self.real_dce_t = self.train_data[:, :, :, :self.input_c_dim]
        self.real_bv_t = self.train_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.real_dce_v = self.val_data[:, :, :, :self.input_c_dim]
        self.real_bv_v = self.val_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_bv_t = self.generator(self.real_dce_t)

        self.real_dceANDbv = tf.concat([self.real_dce_t, self.real_bv_t], 3)
        self.fake_dceANDbv = tf.concat([self.real_dce_t, self.fake_bv_t], 3)
        self.D, self.D_logits = self.discriminator(self.real_dceANDbv, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_dceANDbv, reuse=True)

        self.fake_bv_t_sample = self.sampler(self.real_dce_t)
        self.fake_bv_v_sample = self.sampler(self.real_dce_v)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                                  labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                                  labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.l1_penalty = self.l1_lambda * tf.reduce_mean(tf.abs(self.real_bv_t - self.fake_bv_t))
        self.l1_penalty_v = self.l1_lambda * tf.reduce_mean(tf.abs(self.real_bv_v - self.fake_bv_v_sample))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                             labels=tf.ones_like(self.D_))) + self.l1_penalty

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.bv_t_sum = tf.summary.image('real_vs_fake_bv_train', tf.concat([self.real_bv_t, self.fake_bv_t_sample], 2))
        self.dce_t_ex = tf.concat([self.real_dce_t[:, :, :, 5],
                                   self.real_dce_t[:, :, :, 10],
                                   self.real_dce_t[:, :, :, 25],
                                   self.real_dce_t[:, :, :, 40]], 2)
        self.dce_t_ex = tf.expand_dims(self.dce_t_ex, axis=-1)
        self.dce_t_sum = tf.summary.image('dce_input_train', self.dce_t_ex)

        self.bv_v_sum = tf.summary.image('real_vs_fake_bv_val', tf.concat([self.real_bv_v, self.fake_bv_v_sample], 2))
        self.dce_v_ex = tf.concat([self.real_dce_v[:, :, :, 5],
                                   self.real_dce_v[:, :, :, 10],
                                   self.real_dce_v[:, :, :, 25],
                                   self.real_dce_v[:, :, :, 40]], 2)
        self.dce_v_ex = tf.expand_dims(self.dce_v_ex, axis=-1)
        self.dce_v_sum = tf.summary.image('dce_input_val', self.dce_v_ex)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.l1_penalty_sum = tf.summary.scalar("l1_penalty", self.l1_penalty)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.l1_penalty_sum_v = tf.summary.scalar("l1_penalty_v", self.l1_penalty_v)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train_graph(self, lr=0.0002, beta1=0.5, epochs=100):
        d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.bv_t_sum,
                                       self.dce_t_sum, self.bv_v_sum,
                                       self.dce_v_sum, self.d_loss_fake_sum,
                                       self.g_loss_sum, self.l1_penalty_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1

        if self.load_checkpoint is True:
            self.load_model(self.checkpoint_dir)

        for epoch in range(epochs):
            for idx in range(self.train_batches):
                t_data = next(self.train_data_gen)
                train_sample = np.concatenate((t_data[0], t_data[1]), axis=-1)

                v_data = next(self.valid_data_gen)
                valid_sample = np.concatenate((v_data[0], v_data[1]), axis=-1)

                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.train_data: train_sample})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.train_data: train_sample,
                                                                                 self.val_data: valid_sample})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.train_data: train_sample,
                                                                                 self.val_data: valid_sample})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.train_data: train_sample})
                errD_real = self.d_loss_real.eval({self.train_data: train_sample})
                errG = self.g_loss.eval({self.train_data: train_sample})

                print(errD_fake, errD_real, errG)

                counter += 1

                if np.mod(counter, 500) == 2:
                    self.save_model(self.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = leaky_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = leaky_relu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = leaky_relu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = leaky_relu(self.d_bn3(conv2d(h2, self.df_dim*8, stride=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(leaky_relu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(leaky_relu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(leaky_relu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(leaky_relu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(leaky_relu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(leaky_relu(e6), self.gf_dim*8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(leaky_relu(e7), self.gf_dim*8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = tpconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim*8],
                                                     name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 = tf.concat([self.g_bn_d1(self.d1), e7], 3)

            self.d2, self.d2_w, self.d2_b = tpconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim*8],
                                                     name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 = tf.concat([self.g_bn_d2(self.d2), e6], 3)

            self.d3, self.d3_w, self.d3_b = tpconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim*8],
                                                     name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 = tf.concat([self.g_bn_d3(self.d3), e5], 3)

            self.d4, self.d4_w, self.d4_b = tpconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim*8],
                                                     name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = tpconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim*4],
                                                     name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = tpconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim*2],
                                                     name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = tpconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim],
                                                     name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = tpconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim],
                                                     name='g_d8', with_w=True)

            return tf.nn.tanh(self.d8)

    def sampler(self, image):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(leaky_relu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(leaky_relu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(leaky_relu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(leaky_relu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(leaky_relu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(leaky_relu(e6), self.gf_dim*8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(leaky_relu(e7), self.gf_dim*8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = tpconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim*8],
                                                     name='g_d1', with_w=True)
            self.d1, self.d1_w, self.d1_b = tpconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim * 8],
                                                     name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 = tf.concat([self.g_bn_d1(self.d1), e7], 3)

            self.d2, self.d2_w, self.d2_b = tpconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8],
                                                     name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 = tf.concat([self.g_bn_d2(self.d2), e6], 3)

            self.d3, self.d3_w, self.d3_b = tpconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8],
                                                     name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 = tf.concat([self.g_bn_d3(self.d3), e5], 3)

            self.d4, self.d4_w, self.d4_b = tpconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim*8],
                                                     name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = tpconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim*4],
                                                     name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = tpconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim*2],
                                                     name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = tpconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim],
                                                     name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = tpconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim],
                                                     name='g_d8', with_w=True)

            return tf.nn.tanh(self.d8)

    def save_model(self, checkpoint_dir, step):
        model_name = "cGAN.model"
        model_dir = "%s_%s_%s" % ('test', self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_model(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

    def predict_on_graph(self, test_gen=None, file_name=None):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.load_model(self.checkpoint_dir)

        n_samples = len(test_gen)
        data = test_gen.generate()

        f = h5py.File(file_name, 'w')
        for i in range(n_samples):
            sample = next(data)
            sample_images = sample[0]
            sample_images = sample_images[np.newaxis, :, :, :]
            blank_bv = np.zeros((self.batch_size, self.image_size, self.image_size, self.output_c_dim))
            sample_images = np.concatenate((sample_images, blank_bv), axis=-1)
            samples = self.sess.run(self.fake_bv_t_sample, feed_dict={self.train_data: sample_images})
            f.create_dataset(str(i), data=samples)

        f.close()


def main(_):
    train_imgs = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_train_images.h5'
    train_annos = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_train_annos.h5'
    valid_imgs = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_validation_images.h5'
    valid_annos = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_validation_annos.h5'
    test_imgs = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_test_images_1.h5'
    # ckpt_dir = r'Y:\Prostate Brachytherapy\ABTI_clean_files\ismrm_unet_cgan_run7'
    ckpt_dir = './abti_ckpt'
    # preds_file_name = r'Y:\Prostate Brachytherapy\ABTI_clean_files\abti_test_preds_1.h5'

    load_ckpt = False
    # load_ckpt = True
    dl_action = 'train'
    cgan_type = 'cgan'
    epochs = 100
    # dl_action = 'test'

    train_data = FCN2DDatasetGenerator(train_imgs,
                                       annos_hdf5_path=train_annos,
                                       flip_horizontal=True,
                                       shuffle_data=True,
                                       rounds=1,
                                       batch_size=1,
                                       subset='train',
                                       normalization='samplewise_negpos_xy',
                                       apply_aug=False)

    val_data = FCN2DDatasetGenerator(valid_imgs,
                                     annos_hdf5_path=valid_annos,
                                     shuffle_data=True,
                                     batch_size=1,
                                     subset='validation',
                                     normalization='samplewise_negpos_xy',
                                     apply_aug=False)

    test_data = FCN2DDatasetGenerator(test_imgs,
                                      shuffle_data=True,
                                      batch_size=1,
                                      subset='test',
                                      normalization='samplewise_negpos_xy',
                                      apply_aug=False)

    with tf.Session() as sess:
        if cgan_type == 'cgan':
            model = CGAN(sess,
                         image_size=256,
                         batch_size=1,
                         output_size=256,
                         gf_dim=64,
                         df_dim=64,
                         l1_lambda=100,
                         input_c_dim=60,
                         output_c_dim=1,
                         checkpoint_dir=ckpt_dir,
                         load_checkpoint=load_ckpt,
                         train_data_gen=train_data,
                         valid_data_gen=val_data)
        elif cgan_type == 'cascaded_cgan':
            model = CascadedCGAN(sess,
                                 image_size=256,
                                 batch_size=1,
                                 output_size=256,
                                 gf_dim=64,
                                 df_dim=64,
                                 l1_lambda=100,
                                 input_c_dim=60,
                                 output_c_dim=1,
                                 checkpoint_dir=ckpt_dir,
                                 load_checkpoint=load_ckpt,
                                 train_data_gen=train_data,
                                 valid_data_gen=val_data)
        else:
            ValueError('cgan_type must be either cgan or cascaded_cgan')

        if dl_action == 'train':
            model.train_graph(epochs=epochs)
        else:
            model.predict_on_graph(test_data, preds_file_name)


if __name__ == '__main__':
    tf.app.run()
