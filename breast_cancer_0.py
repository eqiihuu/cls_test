#! /usr/bin/python

# Author: Qi Hu
# Email: qihucn@uw.edu
# ==============================================================================
import os
import time
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf

# --------------------- Settings ---------------------- #
LOAD = 0  # Whether to load existed model
SIZE = 256  # Size of each image
MAX_ITERATION = 100  # Maximum number of iteration
BATCH_SIZE = 100  # Number of images in each batch
TRAIN_BATCH = 9  # Number of train batch
ALL_BATCH = 10  # Number of all batch
IMAGE_NUM = BATCH_SIZE*ALL_BATCH  # Number of images


# Load data --------
def load_data(path):
    print 'Loading images...'
    img_list = os.listdir(path)
    data = np.ndarray((IMAGE_NUM, SIZE, SIZE, 3))

    for i in range(IMAGE_NUM):  #
        img_name = img_list[i]
        img = Image.open(os.path.join(path, img_name))
        img1 = np.asarray(img)
        if (i % 100) == 0:
            print (100.0*i/IMAGE_NUM), '%'
        data[i, :, :, :] = img1
    return data


# Load label ----------
def load_label(d_path, l_path):
    print 'Loading labels...'
    img_list = os.listdir(d_path)
    label_list = pd.read_excel(l_path)
    label = np.ndarray((IMAGE_NUM, 1))
    for i in range(IMAGE_NUM):
        index = img_list[i].split('.')[0]
        diag = label_list[label_list.Filename == index]['Diagnosis'].values[0]
        label[i, 0] = diag
    return label


# Initialize all Weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Initialize all Bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Define convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Define pooling layer
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# --------------------- Load Data ---------------------- #
dir_path = '/Users/qihucn/Documents/EE599/BreastCancer/BreastCancer'
d_path = dir_path+'/data_256'
l_path = dir_path+'/extendDiagnosis.xls'
data = load_data(d_path)
label = load_label(d_path, l_path)
x = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_actual = tf.placeholder(tf.float32, shape=[None, 1])

# ------------------- Define Network -------------------- #
# Conv1
# x_image = tf.reshape(x, [-1, SIZE, SIZE, 3])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

# Conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool(h_conv2)

# Conv3
W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3)+b_conv3)
h_pool3 = max_pool(h_conv3)

# Conv4
W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4)+b_conv4)
h_pool4 = max_pool(h_conv4)

# Conv5
W_conv5 = weight_variable([5, 5, 256, 512])
b_conv5 = bias_variable([512])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5)+b_conv5)
h_pool5 = max_pool(h_conv5)


# Fully Connected 1
W_fc1 = weight_variable([8*8*512, 4096])
b_fc1 = bias_variable([4096])
h_fc1_flat = tf.reshape(h_pool5, [-1, 8*8*512])  # Reshape into 1-dimension
h_fc1 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc1)+b_fc1)
# Dropout 1
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected 2
W_fc2 = weight_variable([4096, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
# Dropout 2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Softmax
W_fc3 = weight_variable([512, 4])
b_fc3 = bias_variable([4])
y_predict = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

saver = tf.train.Saver()

print "Start training ..."
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))  # Cross entropy
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # Gradient descent
# [ tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=Loss) ]
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
if LOAD == 0:
    # --------------------- Training ---------------------- #
    for i in range(MAX_ITERATION):
        if i >= TRAIN_BATCH:
            start = (i % TRAIN_BATCH)*BATCH_SIZE
            end = ((i+1) % TRAIN_BATCH)*BATCH_SIZE
        else:
            start = i*BATCH_SIZE
            end = (i+1)*BATCH_SIZE
        batch_data = data[start:end]
        batch_label = label[start:end]
        train_step.run(feed_dict={x: batch_data, y_actual: batch_label, keep_prob: 0.5})
        if (i % 10) == 0:
            train_acc = accuracy.eval(feed_dict={x: batch_data, y_actual: batch_label, keep_prob: 1.0})
            print 'Batch %d, training accuracy %.3f' % (i, train_acc)
        else:
            print 'Batch %d' % i

    # ----------------- Save teh model ------------------- #
    print "Save model ..."
    t1 = time.time()
    save_path = dir_path+"/model_%.3f.ckpt" % t1
    saver.save(sess, save_path)
else:
    save_path = dir_path+"/model.ckpt"
    saver.restore(sess, save_path)
# --------------------- Testing ---------------------- #
print "Start testing ..."
batch_data = data[BATCH_SIZE*TRAIN_BATCH:BATCH_SIZE*ALL_BATCH]
batch_label = label[BATCH_SIZE*TRAIN_BATCH:BATCH_SIZE*ALL_BATCH]
test_acc = accuracy.eval(feed_dict={x: batch_data, y_actual: batch_label, keep_prob: 1.0})
print 'Test accuracy %.3f' % test_acc
# --------------------- The End ---------------------- #
