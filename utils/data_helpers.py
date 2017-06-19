#!coding=utf-8
import numpy as np
import logging
import random
import os
import codecs

UNK = 'NULL'
ZERO = 'ZERO'
UNK_REG = 'N/A'
embedding_size = None
sentence_length = None
reg_length = None


def read_word_lookup_table(lookup_table_file):
    # Find the word in word2vec 300 table
    lt = []
    word2id = {}
    # for zero padding
    word2id[ZERO] = 0
    lt.append([0 for _ in range(embedding_size)])
    # for zero padding
    # word2id[UNK] = 0
    # lt.append([random.random() for _ in range(embedding_size)])

    f = open(lookup_table_file, 'r')
    f.readline()
    for line in f:
        line = line.strip().decode('utf-8')
        line = line.split()
        word = line[0]
        vec = [float(t) for t in line[1:len(line)]]
        lt.append(vec)
        word2id[word] = len(word2id)
    # for unknown word
    word2id[UNK] = len(word2id)
    lt.append([random.random() for _ in range(embedding_size)])

    f.close()
    id2word = {v: k for k, v in word2id.items()}
    return (lt, word2id, id2word)


def read_label(label_id_file):
    label_id = {}
    id_to_label = []
    V = 0
    f = open(label_id_file, 'r')
    for line in f:
        line = line.strip().decode('utf-8')
        label_id[line] = V
        id_to_label.append(line)
        V += 1
    return (label_id, id_to_label)


def encode_sentence(sent, word2id):
    words = sent.split(" ")
    unigrams = np.zeros(shape=[sentence_length])
    for i in range(len(words)):
        if i >= sentence_length:
            break
        if words[i] not in word2id:
            unigrams[i] = word2id[UNK]
        else:
            unigrams[i] = word2id[words[i]]
    return unigrams


def encode_and_record_reg(reg_str, reg2id, id2reg):
    reg_tags = reg_str.split(' ')
    reg_ids = np.zeros(shape=[reg_length])
    for i in range(len(reg_tags)):
        if i >= reg_length:
            break
        if reg_tags[i] in reg2id:
            reg_ids[i] = reg2id[reg_tags[i]]
        else:
            reg2id[reg_tags[i]] = len(reg2id)
            id2reg.append(reg_tags[i])
            reg_ids[i] = reg2id[reg_tags[i]]
    return reg_ids


def encode_reg(reg_str, reg2id):
    reg_tags = reg_str.split(" ")
    reg_ids = np.zeros(shape=[reg_length])
    for i in range(len(reg_tags)):
        if i >= reg_length:
            break
        if reg_tags[i] in reg2id:
            reg_ids[i] = reg2id[reg_tags[i]]
        else:
            reg_ids[i] = reg2id[UNK_REG]
    return reg_ids


def encode_y(task, label2id):
    tmp = np.zeros(shape=[len(label2id)])
    tmp[label2id[task]] = 1
    return tmp


def read_train(feature_file, label2id, word2id):
    utterances = []
    regs = []
    early_stops = []
    y = []

    reg2id = {}
    id2reg = []
    reg2id[UNK_REG] = 0
    id2reg.append(UNK_REG)

    with open(feature_file,'r') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            line = line.split('\t')

            y.append(encode_y(line[0], label2id))
            early_stops.append( min(len(line[1].split(" ")), sentence_length))
            utterances.append(encode_sentence(line[1], word2id))
            regs.append(encode_and_record_reg(line[2], reg2id, id2reg))

    logging.info("regex size: {}".format(len(reg2id)))
    return (utterances,early_stops,regs,y,reg2id,id2reg)


def read_dev(feature_file, label2id, word2id, reg2id):
    utterances = []
    regs = []
    y = []

    with open(feature_file,'r') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            line = line.split('\t')

            y.append(encode_y(line[0], label2id))
            utterances.append(encode_sentence(line[1],word2id))
            regs.append(encode_reg(line[2], reg2id))

    return (utterances, regs, y)


def export(export_dir,word2id,reg2id,label2id):
  logging.info("saving vocab.dir")
  with codecs.open(os.path.join(export_dir,"vocab.dir"),'w','utf-8') as f:
    f.write(u"{0} {1}\n".format(word2id[UNK],sentence_length))
    for key,value in word2id.items():
        f.write(u"{0} {1}\n".format(key,value))
  logging.info("saving reg.dir")
  with codecs.open(os.path.join(export_dir,"reg.dir"),'w','utf-8') as f:
    f.write(u"{0} {1}\n".format(reg2id[UNK_REG],reg_length))
    for key,value in reg2id.items():
        f.write(u"{0} {1}\n".format(key,value))
  logging.info("saving label.dir")
  with codecs.open(os.path.join(export_dir,"label.dir"),'w','utf-8') as f:
    for key,value in label2id.items():
        f.write(u"{0} {1}\n".format(key,value))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def dump_to_pkl(obj,fname):
    logging.info("writing to pkl")
    with open(fname,'wb') as f:
        np.pickle.dump(obj, f)


def load_from_pkl(fname):
    logging.info("loading from  pkl")
    with open(fname,'rb') as f:
        return np.pickle.load(f)


def extract_dev_diff(dev_file, predictions, id2label, out_file):
    with open(dev_file) as input, codecs.open(out_file, "w", encoding="utf-8") as out:
      for line, pre in zip(input, predictions):
        line = line.strip().decode('utf-8')
        l = line.split('\t')

        id = pre.astype(np.int64)
        if l[0] != id2label[id]:
            out.write(line + u"\t" + id2label[id] + u"\n")
