#!coding=utf-8
import numpy as np
import logging
import random
import os
import codecs

VDS_LENGTH = 12
vds_size = None
REG_SIZE = 166

UNK = 'NULL'
ZERO = 'ZERO'
UNK_REG = 'N/A'
embedding_size = None
sentence_length = None
reg_length = None


# Function:
#     Find the word in word2vec 300 table
def read_word_lookup_table(lookup_table_file):
    lt = []
    word2id = {}
    # Zero padding
    word2id[ZERO] = 0
    lt.append([0 for _ in range(embedding_size)])
    # # Random padding
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
    return lt, word2id, id2word


# Function:
#     Add VDS word-level features (valid_tag) to data_path
# Input:
#     map_path: path of the word2tag mapping file
#     tag_path: path of the tag-id mapping file
# Output:
#     word2tag_dict: the word-tag mapping dictionary
#     each key is a word(raw_str) and the value is a list of its tags
def read_word2tag(word_path, tag_path):
    word2tag_dict = {}
    word2tagid_dict = {}
    word2id = {}
    id2tagvect = []
    tag2id_dict, id2tag_list = read_tag2id(tag_path)
    f = open(word_path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        line = lines[i]
        raw_str = unicode(line.split('\t')[0], 'utf-8')
        tags = line.split('\t')[1].strip('\n')
        tag_list = tags.split(' ')
        for j in range(len(tag_list), VDS_LENGTH):
            tag_list.append('NULL')
        word2tag_dict[raw_str] = tag_list
        word2tagid_dict[raw_str] = []
        word2id[raw_str] = i
        for tag in tag_list:
            tagid = tag2id_dict[tag]
            word2tagid_dict[raw_str].append(tagid)
        id2tagvect.append(word2tagid_dict[raw_str])
    return word2id, word2tagid_dict, id2tagvect


# Function:
#     Read the tag2id mapping file
# Input:
#     tag_path: path of the file that contains all the tags
# Output:
#      tag2id_dict: the dictionary that maps all tags to ids
#      id2tag_list: the list of all tags
def read_tag2id(tag_path):
    tag2id_dict = {'NULL': 0}
    id2tag_list = ['NULL']
    f = open(tag_path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        line = lines[i]
        tag = line.split(' ')[1]
        tag2id_dict[tag] = i+1
        id2tag_list.append(tag)
    return tag2id_dict, id2tag_list


# Function:
#     Read label
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
    return label_id, id_to_label


# Function:
#     Read data
def read_data(feature_file, label2id, word2id, reg2id):
    utterances = []
    vds = []
    regs = []
    y = []
    # print word2id
    with open(feature_file, 'r') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            line = line.split('\t')
            y.append(encode_y(line[0], label2id))
            utterances.append(encode_sentence(line[1], word2id))
            vds.append(encode_multihot_vds(line[3]))
            regs.append(encode_and_record_reg(line[2], reg2id))
        # print vds
    return utterances, vds, regs, y


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


def encode_and_multihot_reg(reg_str, reg2id):
    reg_tags = reg_str.split(' ')
    reg_ids = np.zeros(shape=[REG_SIZE])
    for i in range(len(reg_tags)):
        if reg_tags[i] in reg2id:
            index = reg2id[reg_tags[i]]
            reg_ids[index] = 1
        else:
            reg2id[reg_tags[i]] = len(reg2id)
            reg_ids[len(reg2id)-1] = 1
    return reg_ids


def encode_and_record_reg(reg_str, reg2id):
    reg_tags = reg_str.split(' ')
    reg_ids = np.zeros(shape=[reg_length])
    for i in range(len(reg_tags)):
        if i >= reg_length:
            break
        if reg_tags[i] in reg2id:
            reg_ids[i] = reg2id[reg_tags[i]]
        else:
            reg2id[reg_tags[i]] = len(reg2id)
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


def encode_vds(vds_str):
    words = vds_str.split('|')
    tags = np.zeros(shape=[sentence_length, VDS_LENGTH])
    for i in range(len(words)):
        if i >= sentence_length:
            break
        tids = words[i].split(' ')
        for j in range(len(tids)):
            index = int(tids[j])
            tags[i, j] = index
    # print tags.shape
    return tags


def encode_multihot_vds(vds_str):
    words = vds_str.split('|')
    tags = np.zeros(shape=[sentence_length, vds_size])
    for i in range(len(words)):
        if i >= sentence_length:
            break
        tids = words[i].split(' ')
        for j in tids:
            index = int(j)
            if index >= vds_size:
                continue
            else:
                tags[i, index] = 1
    # print tags.shape
    return tags


# Function:
#     Get the batch data & label
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
