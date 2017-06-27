import numpy as np
import tensorflow as tf
import os

import data_helper as dh
from model.cnn import CNN
from train import train_cnn as train_cnn

__author__ = 'Qi Hu'
__date__ = 'June 15, 2017'
__email__ = 'qihu@mobvoi.com'


def test_cnn(root_path, step_max_devacc, max_devacc):
    dropout = 0.5
    l2_reg = 0.0001
    learning_rate = 0.001
    gpu = 0
    word_lookup_file = './data/word_vectors_pruned_300.txt'
    label_id_file = './data/nlu.label_id.txt'
    test_feature_file = './data_vds_id_test'

    sentence_length = 20
    reg_length = 8
    word_embed = 300
    dh.embedding_size = word_embed
    dh.reg_length = reg_length
    dh.sentence_length = sentence_length
    print('Reading word lookup table...')
    id2vect, word2id, id2word = dh.read_word_lookup_table(word_lookup_file)
    id2vect = np.asarray(id2vect, dtype=np.float32)

    print('Reading label id...')
    label2id, id2label = dh.read_label(label_id_file)

    print('Reading test data...')
    reg2id = {'N/A': 0}
    test_word, test_vds, test_reg, test_y = dh.read_data(test_feature_file, label2id, word2id, reg2id)

    vbs_size = 308
    vocab_size = len(word2id)
    reg_size = len(reg2id)
    num_class = len(label2id)

    cnn = CNN(num_class, id2vect, gpu, l2_reg, dropout, learning_rate, vocab_size, vbs_size, reg_size)
    test_acc = cnn.test(root_path, step_max_devacc, max_devacc, test_word, test_vds, test_reg, test_y)
    print('Test accuracy: %.3f' % test_acc)


if __name__ == '__main__':
    root_path, step_max_devacc, max_devacc = train_cnn()
    test_cnn(root_path, step_max_devacc, max_devacc)
