import numpy as np
import tensorflow as tf

import utils.data_helpers as dh
from model.cnn import CNN

__author__ = 'Qi Hu'
__date__ = 'June 15, 2017'
__email__ = 'qihu@mobvoi.com'


def train_cnn():
    sentence_length = 20
    reg_length = 8
    word_embed = 300
    reg_embed = 250
    dropout = 0.5
    l2_reg = 0.0001
    batch_size = 100
    learning_rate = 0.001
    checkpoint_step = 300
    epoch_num = 2
    version = 0
    filter_sizes = [2, 3, 4]
    filter_num = 64
    gpu = 0
    model_name = 'model_0616_'
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
    train_x, train_stops, train_regs, train_y, reg2id, id2reg = dh.read_train(train_feature_file, label2id, word2id)

    print('Reading dev data...')
    dev_x, dev_regs, dev_y = dh.read_dev(dev_feature_file, label2id, word2id, reg2id)

    vocab_size = len(word2id)
    reg_size = len(reg2id)
    num_class = len(label2id)

    cnn = CNN(id2vect, gpu, dropout, learning_rate, vocab_size, reg_size,
              sentence_length, word_embed, reg_length, reg_embed,
              num_class, filter_sizes, filter_num)
    dev_acc, dev_pred = cnn.train(dropout, checkpoint_step,
                                  batch_size, epoch_num, model_name, version,
                                  train_x, train_regs, train_y, dev_x, dev_regs, dev_y,
                                  )
    print('Dev accuracy: ', dev_acc)
if __name__ == '__main__':
    train_cnn()
