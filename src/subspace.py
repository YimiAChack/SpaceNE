#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from utils import *


class SpaceNE(object):
    def __init__(self, input_X, num_community, input_dim, params):
        tf.reset_default_graph()
        self.divide_point_u = params["SpaceNE"]["divide_point_u"]
        self.X = input_X
        self.alpha = tf.constant(params["SpaceNE"]["alpha"], 'float32')
        self.lambda_ = tf.constant(params["SpaceNE"]["lambda_"], 'float32')
        self.num_community = num_community
        self.tensor_graph = tf.Graph()
        self.epoch = params["SpaceNE"]["epoch"]
        self.output_dim = input_dim - params["SpaceNE"]["dimension_reduce"]
        self.save_path = params["base_path"] + params["SpaceNE"]["model_save_path"]
        self.max_epoch = params["SpaceNE"]["max_epoch"]
        self.U = tf.Variable(np.random.randn(self.num_community, input_dim, self.output_dim).astype(np.float32),name='W')

    def train(self):
        if self.save_path is None:
            print("model save_path is None!")
            return

        print("SpaceNE start training... ")

        loss = self.within_class_dis() + self.alpha * self.between_class_dis() + self.lambda_ * self.reg_U()

        objective_loss = tf.clip_by_value(loss, 1e-8, float('inf') - 1)
        # optimization setup
        train_step = tf.train.AdamOptimizer(0.01).minimize(objective_loss)
        saver = tf.train.Saver()

        # fit the model
        min_loss = float('inf')
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            for n in range(self.max_epoch):
                sess.run(train_step)
                if (n + 1) % self.epoch == 0:
                    current_loss = sess.run(loss)
                    print('iter %i, %f' % (n + 1, current_loss))
                    if current_loss < min_loss:
                        min_loss = current_loss
                        save_path_sess = saver.save(sess, self.save_path)
                        print("save model to..", save_path_sess)

    def within_class_dis(self):
        sim_inner = tf.constant(0.0, 'float32')
        for i in range(self.num_community):
            part1 = tf.matmul(tf.transpose(self.U[i]), tf.transpose(self.X[i]))
            part2 = tf.matmul(self.X[i], self.U[i])
            sim_inner -= tf.trace(tf.matmul(part1, part2))
        return sim_inner

    def between_class_dis(self):
        sim_external = tf.constant(0.0, 'float32')
        for i in range(self.num_community):
            for j in range(i + 1, self.num_community):
                part1_1 = tf.matmul(self.X[i], self.U[i])
                part1_2 = tf.matmul(tf.transpose(self.U[j]), tf.transpose(self.X[j]))
                part1 = tf.matmul(part1_1, part1_2)
                part2 = tf.matmul(self.X[i], tf.transpose(self.X[j]))
                sim_external += tf.norm(tf.subtract(part1, part2), 'euclidean')  # F
        return sim_external

    def reg_U(self):
        reg_U = tf.constant(0.0, 'float32')
        for i in range(self.num_community):
            reg_U += self.rank_approximation(self.U[i])
        return reg_U

    def rank_approximation(self, X):
        divide_point = tf.constant(self.divide_point_u, 'float32')
        s, u, v = tf.svd(X)

        def cal2(m, n):
            return tf.divide(tf.multiply(m, m), tf.multiply(n, 2))

        def cal1(m, n):
            return tf.subtract(m, tf.divide(n, 2))

        res = tf.reduce_sum(tf.where(s > divide_point, cal2(s, divide_point), cal1(s, divide_point)))

        return res

    def load_W(self):
        saver = tf.train.import_meta_graph(self.save_path + '.meta')
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            W = sess.run(tf.get_default_graph().get_tensor_by_name('W:0'))  # the learnt projection matrix
        return W

    '''
    def reconstruction_X(self,W,input_X):
        for i in range(self.num_community):
            W_ = utils.gram_schmidt(W[i])
            input_X[i] = np.dot(np.dot(W_,W_.transpose()),input_X[i].transpose()).transpose()

        rescon_X = X_comm_to_arr(input_X)
        return rescon_X
    '''
