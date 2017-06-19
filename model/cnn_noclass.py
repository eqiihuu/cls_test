import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

import utils.data_helpers as dh

__author__ = 'qihu'
__date__ = 'June 17, 2017'
__email__ = 'qihu@mobvoi.com'

# the vector is randomly initialized each time
use_gpu = 0  # whether to use GPU
dropout_keep = 0.5  # keep probability of dropout layer
learning_rate = 0.001  # initial learning rate
sentence_length = 20  # length of a sentence 20
word_embed = 300  # the embedding length of a word 300
reg_length = 8  # the length of RegEx in in a sentence 8
reg_embed = 250  # the embedding length of a RegEx 250
filter_sizes = [2, 3, 4]  # a list size of conv filters' size
filter_num = 64  # number of filters for a single filter_size
l2_reg = 0.001  # L2 regularization
checkpoint_step = 100
version = 0
batch_size = 100
epoch_num = 2

model_name = 'model_0617_'
word_lookup_file = './data/word_vectors_pruned_300.txt'
label_id_file = './data/nlu.label_id.txt'
train_feature_file = './data/nlu.train.string.cnn_format'
dev_feature_file = './data/nlu.dev.string.cnn_format'

dh.embedding_size = word_embed
dh.reg_length = reg_length
dh.sentence_length = sentence_length

print('Reading word lookup table...')
id2vect, word2id, id2word = dh.read_word_lookup_table(word_lookup_file)
id2vect = np.asarray(id2vect, dtype=np.float32)

print('Reading label id...')
label2id, id2label = dh.read_label(label_id_file)

print('Reading train data...')
train_word, train_stops, train_reg, train_y, reg2id, id2reg = dh.read_train(train_feature_file, label2id, word2id)

print('Reading dev data...')
dev_word, dev_reg, dev_y = dh.read_dev(dev_feature_file, label2id, word2id, reg2id)

vocab_size = len(word2id)
reg_size = len(reg2id)
num_class = len(label2id)

device = (use_gpu and '/gpu:0') or '/cpu:0'

x_word = tf.placeholder(tf.int32, shape=(None, sentence_length))
x_reg = tf.placeholder(tf.int32, shape=(None, reg_length))
y = tf.placeholder(tf.float32, shape=(None, num_class))

# RegEx mapping
W_r = tf.Variable(tf.random_uniform([reg_size, reg_embed], -1.0, 1.0))
reg_vect = tf.nn.embedding_lookup(W_r, x_reg)
reg_norm = tf.reduce_max(reg_vect, 1)  # Max-pooling on first dimension

# Embedding layer for words
embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embed]),
                             trainable=True, name='embeddomg')  # why not just create randomly?
tf.assign(embedding, id2vect)
embedded_word = tf.nn.embedding_lookup(embedding, x_word)
embedded_word_expanded = tf.expand_dims(embedded_word, -1)  # why expanding?

pool_output = []  # save the outputs of pooling layer

# Convolution layer part0
filter_shape = [filter_sizes[0], word_embed, 1, filter_num]
W0 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
b0 = tf.Variable(tf.constant(0.0, shape=[filter_num]), name='b')
conv0 = tf.nn.conv2d(  # convolution operation
    embedded_word_expanded,
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
    embedded_word_expanded,
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
    embedded_word_expanded,
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
h_pool = tf.concat(pool_output, 3)
h_pool_flat = tf.reshape(h_pool, [-1, filter_totalnum])

# Dropout
keep_prob = tf.placeholder('float')
reg_drop = tf.nn.dropout(reg_norm, keep_prob)

# Merge word features and RegEx features
feature = tf.concat([h_pool_flat, reg_drop], 1)

l2_loss = tf.constant(0.0)
# Score and prediction
W_f = tf.Variable(tf.constant(0.0, shape=[filter_totalnum+reg_embed, num_class]), name='W')
b_f = tf.Variable(tf.constant(0.0, shape=[num_class]), name='b')
l2_loss += tf.nn.l2_loss(W_f) + tf.nn.l2_loss(b_f)
score = tf.nn.xw_plus_b(feature, W_f, b_f, name='score')
prob = tf.nn.softmax(score, name='prob')
prediction = tf.argmax(prob, 1, name='prediction')

# Cross-entropy loss
loss = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y)
loss = tf.reduce_mean(loss) + l2_reg * l2_loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy
y_index = tf.argmax(y, 1)
correct = tf.equal(prediction, y_index)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')

# Training
curr_step = 0
batches = dh.batch_iter(list(zip(train_word, train_reg, train_y)), batch_size, epoch_num)
dev_feed_dict = {x_word: dev_word,
                 x_reg: dev_reg,
                 y: dev_y,
                 keep_prob: 1.0}
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for batch in batches:
    if len(batch) == 0:
        continue
    word_batch, reg_batch, y_batch = zip(*batch)
    train_feed_dict = {x_word: word_batch,
                 x_reg: reg_batch,
                 y: y_batch,
                 keep_prob: dropout_keep}
    train_step.run(feed_dict=train_feed_dict)


    curr_step += 1
    if curr_step % checkpoint_step == 0:
        dev_acc = accuracy.eval(dev_feed_dict)
        train_acc = accuracy.eval(feed_dict=train_feed_dict)
        print 'Step %d, Train Accuracy: %.3f' % (curr_step, train_acc)
        print '         Test Accuracy: ', dev_acc
dev_acc = accuracy.eval(dev_feed_dict)
print 'Final Test Accuracy: ', dev_acc
