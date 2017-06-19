import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

import utils.data_helpers as dh

__author__ = 'qihu'
__date__ = 'June 16, 2017'
__email__ = 'qihu@mobvoi.com'


class CNN(object):
    # Define the model
    def __init__(self,
                 num_class,  # number of sentence classes 51
                 id2vect,  # a list which map the word_id into a vector [5850*300],
                           # the vector is randomly initialized each time
                 use_gpu=0,  # whether to use GPU
                 l2_reg=0.0001,  # L2 regularization
                 dropout_keep=0.5,  # keep probability of dropout layer
                 learning_rate=0.001,  # initial learning rate
                 vocab_size=5850,  # number of all words 5850
                 reg_size=166,  # number of all RegEx 166
                 sentence_length=20,  # length of a sentence 20
                 word_embed=300,  # the embedding length of a word 300
                 reg_length=8,  # the length of RegEx in in a sentence 8
                 reg_embed=250,  # the embedding length of a RegEx 250
                 filter_sizes=[3, ],  # a list size of conv filters' size
                 filter_num=64  # number of filters for a single filter_size
                 ):
        self.device = (use_gpu and '/gpu:0') or '/cpu:0'
        self.dropout_keep = dropout_keep

        self.x_word = tf.placeholder(tf.int32, shape=(None, sentence_length))
        self.x_reg = tf.placeholder(tf.int32, shape=(None, reg_length))
        self.y = tf.placeholder(tf.float32, shape=(None, num_class))

        # RegEx mapping
        with tf.device(self.device), tf.variable_scope('reg_map'):
            W = tf.Variable(tf.random_uniform([reg_size, reg_embed], -1.0, 1.0))
            reg_vect = tf.nn.embedding_lookup(W, self.x_reg)
            self.reg_norm = tf.reduce_max(reg_vect, 1)  # Max-pooling on first dimension

        # Embedding layer for words
        with tf.device(self.device), tf.variable_scope('embed_layer'):
            self.embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embed]),
                                         trainable=True, name='embed')  # why not just create randomly?
            tf.assign(self.embedding, id2vect)
            self.embedded_word = tf.nn.embedding_lookup(self.embedding, self.x_word)
            self.embedded_word_expanded = tf.expand_dims(self.embedded_word, -1)  # why expanding?

        pool_output = []  # save the outputs of pooling layer
        # Convolution layer
        for i, filter_size in enumerate(filter_sizes):
            with tf.device(self.device), tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size, word_embed, 1, filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')
                conv = tf.nn.conv2d(  # convolution operation
                    self.embedded_word_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pool = tf.nn.max_pool(h,
                                      ksize=[1, sentence_length-filter_size+1, 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding='VALID',
                                      name='pool'
                                      )
                pool_output.append(pool)
            filter_totalnum = filter_num*len(filter_sizes)
        self.h_pool = tf.concat(pool_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, filter_totalnum])

        # Dropout
        self.dropout_keep = tf.placeholder('float')
        with tf.device(self.device), tf.name_scope('dropout'):
            self.reg_drop = tf.nn.dropout(self.reg_norm, self.dropout_keep)

        # Merge word features and RegEx features
        with tf.device(self.device), tf.name_scope('merge'):
            self.feature = tf.concat([self.h_pool_flat, self.reg_drop], 1)

        l2_loss = tf.constant(0.0)
        # Score and prediction
        with tf.device(self.device), tf.name_scope('output'):
            W = tf.Variable(tf.constant(0.0, shape=[filter_totalnum+reg_embed, num_class]), name='W')
            b = tf.Variable(tf.constant(0.0, shape=[num_class]), name='b')
            l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            self.score = tf.nn.xw_plus_b(self.feature, W, b, name='score')
            self.prob = tf.nn.softmax(self.score, name='prob')
            self.prediction = tf.argmax(self.prob, 1, name='prediction')

        # Cross-entropy loss
        with tf.device(self.device), tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y)
            self.loss = tf.reduce_mean(loss) + l2_reg * l2_loss
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        with tf.device(self.device), tf.name_scope('loss'):
            y_index = tf.argmax(self.y, 1)
            correct = tf.equal(self.prediction, y_index)
            self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')

    # Training process
    def train(self, dropout, check_step, save_step, batch_size, epoch_num, model_name,
              train_word, train_reg, train_y,
              dev_word, dev_reg, dev_y,
              test_word, test_reg, test_y):
        curr_step = 0
        batches = dh.batch_iter(list(zip(train_word, train_reg, train_y)), batch_size, epoch_num, True)
        dev_feed_dict = {self.x_word: dev_word,
                         self.x_reg: dev_reg,
                         self.y: dev_y,
                         self.dropout_keep: dropout}
        test_feed_dict = {self.x_word: test_word,
                         self.x_reg: test_reg,
                         self.y: test_y,
                         self.dropout_keep: dropout}
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        # Training
        for batch in batches:
            if len(batch) == 0:
                continue
            word_batch, reg_batch, y_batch = zip(*batch)
            train_feed_dict = {self.x_word: word_batch,
                         self.x_reg: reg_batch,
                         self.y: y_batch,
                         self.dropout_keep: dropout}
            self.train_step.run(feed_dict=train_feed_dict)
            curr_step += 1
            if curr_step % check_step == 0:
                dev_acc = self.accuracy.eval(dev_feed_dict)
                train_acc = self.accuracy.eval(train_feed_dict)
                print 'Step %d, Train Accuracy: %.03f' % (curr_step, train_acc)
                print '          Dev Accuracy: %.03f'% dev_acc
                if curr_step % save_step == 0:
                    save_model_path = "./save/model_%d_devacc_%.3f.ckpt" % (curr_step/save_step, dev_acc)
                    saver = tf.train.Saver(tf.global_variables())
                    saver_path = saver.save(sess, save_model_path)
        test_acc = self.accuracy.eval(test_feed_dict)
        save_model_path = "./save/model_testacc_%.3f.ckpt" % test_acc
        saver = tf.train.Saver(tf.global_variables())
        saver_path = saver.save(sess, save_model_path)
        print 'Save model to %s' % saver_path
        return dev_acc, test_acc
