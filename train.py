import numpy as np

import utils.data_helpers as dh
from model.cnn import CNN

__author__ = 'Qi Hu'
__date__ = 'June 15, 2017'
__email__ = 'qihu@mobvoi.com'


def train_cnn():

    dropout = 0.5
    l2_reg = 0.0001
    batch_size = 100
    learning_rate = 0.001
    check_step = 300
    save_step = 900
    epoch_num = 30
    gpu = 0
    model_name = 'model.cnn'
    word_lookup_file = './data/word_vectors_pruned_300.txt'
    label_id_file = './data/nlu.label_id.txt'
    train_feature_file = './data/nlu.train.string.cnn_format'
    dev_feature_file = './data/nlu.dev.string.cnn_format'
    test_feature_file = './data/nlu.test.string.cnn_format'

    sentence_length = 20
    reg_length = 8
    word_embed = 300
    reg_embed = 250
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

    print('Reading test data...')
    test_word, test_reg, test_y = dh.read_dev(test_feature_file, label2id, word2id, reg2id)

    vocab_size = len(word2id)
    reg_size = len(reg2id)
    num_class = len(label2id)

    cnn = CNN(num_class, id2vect, gpu, l2_reg, dropout, learning_rate, vocab_size, reg_size)
    dev_acc, test_acc = cnn.train(dropout, check_step, save_step, batch_size, epoch_num, model_name,
                                  train_word, train_reg, train_y,
                                  dev_word, dev_reg, dev_y,
                                  test_word, test_reg, test_y)
    print('Dev accuracy: %.3f' % dev_acc)
    print('Test accuracy: %.3f' % test_acc)
if __name__ == '__main__':
    train_cnn()
