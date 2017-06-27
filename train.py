import numpy as np

import data_helper as dh
from model.cnn_droplater import CNN

__author__ = 'Qi Hu'
__date__ = 'June 15, 2017'
__email__ = 'qihu@mobvoi.com'


def train_cnn():
    word_embed = 300
    reg_embed = 250
    vds_size = 300

    dropout = 0.5
    l2_reg = 0.0001
    batch_size = 100
    learning_rate = 0.001
    check_step = 300
    save_step = 900
    epoch_num = 15
    gpu = 0
    model_name = 'model.cnn'
    word_lookup_file = './data/word_vectors_pruned_300.txt'
    label_id_file = './data/nlu.label_id.txt'
    train_feature_file = './data_vds_id_train'
    dev_feature_file = './data_vds_id_dev'
    test_feature_file = './data_vds_id_test'

    sentence_length = 20
    reg_length = 8
    filter_size = 3
    filter_num = 64

    dh.vds_size = vds_size
    dh.embedding_size = word_embed
    dh.reg_length = reg_length
    dh.sentence_length = sentence_length
    print('Reading word lookup table...')
    id2vect, word2id, id2word = dh.read_word_lookup_table(word_lookup_file)

    print('Reading label id...')
    label2id, id2label = dh.read_label(label_id_file)

    print('Reading data...')
    reg2id = {'N/A': 0}
    train_word, train_vds, train_reg, train_y = dh.read_data(train_feature_file, label2id, word2id, reg2id)
    dev_word, dev_vds, dev_reg, dev_y = dh.read_data(dev_feature_file, label2id, word2id, reg2id)
    test_word, test_vds, test_reg, test_y = dh.read_data(test_feature_file, label2id, word2id, reg2id)
    # print test_vds[0]

    vocab_size = len(word2id)
    reg_size = len(reg2id)
    num_class = len(label2id)

    cnn = CNN(num_class, gpu, l2_reg, learning_rate, vocab_size, vds_size, reg_size, reg_length,
              sentence_length, word_embed, reg_embed, filter_size, filter_num)
    print 'Start Training'
    max_devacc, step_max_devacc, root_path = cnn.train(dropout, check_step, save_step, batch_size, epoch_num, model_name,
                                                       train_word, train_vds, train_reg, train_y,
                                                       dev_word, dev_vds, dev_reg, dev_y,
                                                       test_word, test_vds, test_reg, test_y,
                                                       id2label)
    print('Dev accuracy: %.3f, at step %d' % (max_devacc, step_max_devacc))
    return root_path, step_max_devacc, max_devacc

if __name__ == '__main__':
    train_cnn()
