from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

import utils.data_helpers as dh

__author__ = 'qihu'
__date__ = 'June 16, 2017'
__email__ = 'qihuchn@gmail.com'


class CNN(object):
    # Define the model
    def __init__(self,
                 id2vect,  # a list which map the word_id into a vector [5850*300],
                           # the vector is randomly initialized each time
                 use_gpu=0,  # whether to use GPU
                 dropout_keep=0.5,  # keep probability of dropout layer
                 learning_rate=0.001,  # initial learning rate
                 vocab_size=5850,  # number of all words 5850
                 reg_size=166,  # number of all RegEx 166
                 sentence_length=20,  # length of a sentence 20
                 word_embed=300,  # the embedding length of a word 300
                 reg_length=8,  # the length of RegEx in in a sentence 8
                 reg_embed=250,  # the embedding length of a RegEx 250
                 num_class=51,  # number of sentence classes 51
                 filter_sizes=[2, 3, 4],  # a list size of conv filters' size
                 filter_num=64,  # number of filters for a single filter_size
                 l2_reg=0.001  # L2 regularization
                 ):
        self.device = (use_gpu and '/gpu:0') or '/cpu:0'
        self.dropout_keep = dropout_keep

        self.x_word = tf.placeholder(tf.int32, shape=(None, sentence_length))
        self.x_reg = tf.placeholder(tf.int32, shape=(None, reg_length))
        self.y = tf.placeholder(tf.float32, shape=(None, num_class))

        # RegEx mapping
        W_r = tf.Variable(tf.random_uniform([reg_size, reg_embed], -1.0, 1.0))
        reg_vect = tf.nn.embedding_lookup(W_r, self.x_reg)
        self.reg_norm = tf.reduce_max(reg_vect, 1)  # Max-pooling on first dimension

        # Embedding layer for words
        self.embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embed]),
                                     trainable=True, name='embeddomg')  # why not just create randomly?
        tf.assign(self.embedding, id2vect)
        self.embedded_word = tf.nn.embedding_lookup(self.embedding, self.x_word)
        self.embedded_word_expanded = tf.expand_dims(self.embedded_word, -1)  # why expanding?

        pool_output = []  # save the outputs of pooling layer

        # Convolution layer part0
        filter_shape = [filter_sizes[0], word_embed, 1, filter_num]
        W0 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        b0 = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')
        conv0 = tf.nn.conv2d(  # convolution operation
            self.embedded_word_expanded,
            W0,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv'
        )
        h0 = tf.nn.relu(tf.nn.bias_add(conv0, b0), name='relu')
        pool0 = tf.nn.max_pool(h0,
                              ksize=[1, sentence_length-filter_sizes[0] + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool'
                              )
        pool_output.append(pool0)

        # Convolution layer part1
        filter_shape = [filter_sizes[1], word_embed, 1, filter_num]
        W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        b1 = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')
        conv1 = tf.nn.conv2d(  # convolution operation
            self.embedded_word_expanded,
            W1,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv'
        )
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name='relu')
        pool1 = tf.nn.max_pool(h1,
                              ksize=[1, sentence_length - filter_sizes[1] + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool'
                              )
        pool_output.append(pool1)

        # Convolution layer part2
        filter_shape = [filter_sizes[2], word_embed, 1, filter_num]
        W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        b2 = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')
        conv2 = tf.nn.conv2d(  # convolution operation
            self.embedded_word_expanded,
            W2,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv'
        )
        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu')
        pool2 = tf.nn.max_pool(h2,
                              ksize=[1, sentence_length - filter_sizes[2] + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name='pool'
                              )
        pool_output.append(pool2)

        filter_totalnum = filter_num*len(filter_sizes)
        self.h_pool = tf.concat(pool_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, filter_totalnum])

        # Dropout
        self.reg_drop = tf.nn.dropout(self.reg_norm, self.dropout_keep)

        # Merge word features and RegEx features
        self.feature = tf.concat([self.h_pool_flat, self.reg_drop], 1)

        l2_loss = tf.constant(0.0)
        # Score and prediction
        W_f = tf.Variable(tf.constant(0.0, shape=[filter_totalnum+reg_embed, num_class]), name='W')
        b_f = tf.Variable(tf.constant(0.0, shape=[num_class]), name='b')
        l2_loss += tf.nn.l2_loss(W_f) + tf.nn.l2_loss(b_f)
        self.score = tf.nn.xw_plus_b(self.feature, W_f, b_f, name='score')
        self.prob = tf.nn.softmax(self.score, name='prob')
        self.prediction = tf.argmax(self.prob, 1, name='prediction')

        # Cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)
        self.loss = tf.reduce_mean(loss) + l2_reg * l2_loss
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        y_index = tf.argmax(self.y, 1)
        correct = tf.equal(self.prediction, y_index)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')

    # Training process
    def train(self, dropout, checkpoint_step, batch_size, epoch_num, model_name, version,
              train_word, train_reg, train_y, dev_word, dev_reg, dev_y):
        curr_step = 0
        batches = dh.batch_iter(list(zip(train_word, train_reg, train_y)), batch_size, epoch_num)
        dev_feed_dict = {self.x_word: dev_word,
                         self.x_reg: dev_reg,
                         self.y: dev_y,
                         self.dropout_keep: dropout}
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        # Training
        for batch in batches:
            if len(batch) == 0:
                continue
            word_batch, reg_batch, y_batch = zip(*batch)
            feed_dict = {self.x_word: word_batch,
                         self.x_reg: reg_batch,
                         self.y: y_batch,
                         self.dropout_keep: dropout}
            self.train_step.run(feed_dict=feed_dict)
            curr_step += 1
            if curr_step % checkpoint_step == 0:
                self.accuracy.run([self.accuracy, self.prediction], dev_feed_dict)
        acc, predictions = self.accuracy.run([self.accuracy, self.prediction], dev_feed_dict)
        export_model_path = './export/%s' % model_name
        if self.device == '/cpu:0':
            saver = tf.train.Saver(tf.global_variables())
            model_exporter = exporter.Exporter(saver)
            named_tensor_binding = {"input_x": self.x_in, "input_reg": self.x_reg,
                                    "classes": self.predictions, "scores": self.probs}
            signature = exporter.generic_signature(named_tensor_binding)
            signatures = {"generic": signature}
            model_exporter.init(sess.graph.as_graph_def(), named_graph_signatures=signatures)
            model_exporter.export(export_model_path, tf.constant(version), sess)
        return acc, predictions
