# coding=utf-8

import os
import random
import json
from utils import vds_caller as vc


# Function:
#     Get the cache file for VDS of [data_path]
def analyse(data_path):
    f = open(data_path)
    lines = f.readlines()
    f.close()
    i = 0
    for line in lines[0:2]:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        print i, domain, sentence
        i += 1
        anno = vc.get_annotations(sentence, domain)
    f.close()


# Function:
#     Merge two files
def merge(src_path, dest_path):
    f_src = open(src_path)
    content = f_src.readlines()
    f_src.close()
    f_dest = open(dest_path, 'a')
    f_dest.write(content)
    f_dest.close()
    print 'Merge %s to %s' % (src_path, dest_path)


# Function:
#     Split a file of data into two part, you can specify the percentage
# Input:
#     data_path: the path of the data file
#     percentage: the percentage of the first part
# Output: (No return value)
#     save the two split parts into two new files (old_split_data & new_split_data)
def split_data(data_path, percent):
    f = open(data_path)
    lines = f.readlines()
    random.shuffle(lines)
    line_num = len(lines)
    old_num = int(line_num*percent)
    old_file = open('old_split_data', 'w')
    new_file = open('new_split_data', 'w')
    for i in range(0, old_num):
        old_file.write(lines[i])
    for i in range(old_num, line_num):
        new_file.write(lines[i])
    old_file.close()
    new_file.close()


# Get the tag list of a sentence and the number of occurance
# If there is already some data in the file, load the file before start getting new tag_list
def get_taglist(data_path):
    tag_list = []
    tag_dict = {}
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = open('tag2id_list.txt', 'r')
    old_list = f.readlines()
    f.close()
    for i in old_list:
        tag = i.split(' ')[1]
        num = int(i.split(' ')[2])
        tag_list.append(tag)
        tag_dict[tag] = num
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        qa_json = vc.get_anno_json(sentence, domain)
        # print qa_json
        try:
            anno = qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']
            for i in anno:
                if 'valid_data_type' in i['value']:
                    tag = i['value']['valid_data_type']
                    # print tag
                    if tag_dict.has_key(tag):
                        tag_dict[tag] = int(tag_dict[tag])+1
                    else:
                        tag_dict[tag] = 1
                    if tag not in tag_list:
                        tag_list.append(tag)
        except Exception as ex:
            print 'error'

    f = open('tag2id_list.txt', 'w')
    items = tag_dict.items()
    items = sorted(items, lambda x, y: cmp(x[1], y[1]), reverse=True)
    for i in range(len(items)):
        tag = items[i][0]
        num = items[i][1]
        s = '%d %s %d \n' % (i, tag, num)
        f.write(s)
        print s
    f.close()


# Function:
#     Get the combined statistic information of train/dev/test dataset
# Input:
#     file_prefix: prefix of all four files (the suffix is shown below)
# Output: (No return value)
#     save the combined statistics to a file called [file_prefix]_combine.txt
def get_combine(file_prefix):
    train_path = file_prefix + '_train.txt'
    dev_path = file_prefix + '_dev.txt'
    test_path = file_prefix + '_test.txt'
    all_path = file_prefix + '_all.txt'
    train_tag_list, train_tag_dict = read_list(train_path)
    dev_tag_list, dev_tag_dict = read_list(dev_path)
    test_tag_list, test_tag_dict = read_list(test_path)
    all_tag_list, all_tag_dict = read_list(all_path)
    items = all_tag_dict.items()
    items = sorted(items, lambda x, y: cmp(x[1], y[1]), reverse=True)
    f = open(file_prefix+'_combine.txt', 'w')
    for i in range(len(items)):
        item = items[i]
        tag = item[0]
        num = item[1]
        train = train_tag_dict[tag]
        try:
            test = test_tag_dict[tag]
        except Exception:
            test = 0
        try:
            dev = dev_tag_dict[tag]
        except Exception:
            dev = 0
        s = '%d\t%d\t%d\t%d\t%s\n' %(num, train, dev, test, tag)
        f.write(s)
        print s
    f.close()


# Function:
#     Read tag_list from a file
# Input:
#     file_path: path of the file to be read
# Output:
#     tag_list: a list of all tags
#     tag_dict: a dictionary of the tags and their numbers
def read_list(file_path):
    tag_dict = {}
    tag_list = []
    f = open(file_path)
    lines = f.readlines()
    f.close()
    for line in lines:
        tag = line.split(' ')[1]
        num = int(line.split(' ')[2])
        if tag not in tag_list:
            tag_list.append(tag)
            tag_dict[tag] = num
        else:
            tag_dict[tag] += 1
    return tag_list, tag_dict


# Function:
#     Add VDS features (valid_tag) to data_path
# Input:
#     data_path: path of the data to be read
#     data_set: which set to process (train/dev/test)
# Output: (no return value)
#     Save the new data to a new file
def add_vds(data_path, data_set):
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = open('data_vds_' + data_set, 'w')
    index = 0
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        qa_json = vc.get_anno_json(sentence, domain)
        # print qa_json
        try:
            tags = ''
            tag_num = 0
            anno = qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']
            for i in anno:
                if 'valid_data_type' in i['value']:
                    tag = i['value']['valid_data_type']
                    tags = tags + tag + ' '
                    tag_num += 1
            line = line[:-1] + '\t' + str(tags[:-1])+'\n'
            index += 1
            print index, tag_num, sentence
        except Exception as ex:
            print 'error'
        f.write(line)
    f.close()
    print 'Number of sentences with vde feature: ', index


if __name__ == '__main__':
    data_set = 'train'
    data_path = '/home/qihu/PycharmProjects/cls_test/data/nlu.' +data_set+ '.string.cnn_format'
    # analyse(data_path)  # get the tag list of data
    # split_data(data_path, 0.5)
    # get_taglist(data_path)
    # analyse(data_path)
    # get_combine('tag2id_list')
    add_vds(data_path, data_set)
