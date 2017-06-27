#  coding=utf-8

import codecs
import random
import json
import numpy as np
import urllib
# from utils import vds_caller as vc

VDS_LENGTH = 12
VDS_SIZE = 308

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
            print 'Error'

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
#     Add VDS sentence-level features (valid_tag) to data_path
# Input:
#     data_path: path of the data to be read
#     data_set: which set to process (train/dev/test)
# Output: (no return value)
#     Save the new data to a new file
def add_sentence_vds(data_path, data_set):
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


# Function:
#     Get the mapping file between all words and their tags
# Input:
#     data_path: the path of raw input data
# Output: (No return value)
#     Save the mapping between each word and its tags into a file
def get_word2tag_map(data_path):
    longest = 0
    word_dict = {}
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = codecs.open('word2tag.txt', 'w', encoding='utf-8')
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        qa_json = vc.get_anno_json(sentence, domain)
        # print qa_json
        try:
            anno = qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']
            for i in anno:
                raw_str = i['raw_str']
                if 'valid_data_type' in i['value']:
                    tag = i['value']['valid_data_type']
                    if not word_dict.has_key(raw_str):
                        word_dict[raw_str] = [tag, ]
                        # print raw_str,tag
                    elif tag not in word_dict[raw_str]:
                        word_dict[raw_str].append(tag)
                        # print raw_str, tag
        except Exception:
            print 'Warning: Can\'t find VDS for %s' % raw_str
    items = word_dict.items()
    for item in items:
        longest = max(longest, len(item[1]))
        line = item[0] + '\t'
        # print item
        for i in range(0, len(item[1])):
            line = line + item[1][i]+' '
        # print line
        line = line[:-1]+'\n'
        f.write(line)
    f.close()
    print longest


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
    tag2id_dict, id2tag_list = read_tag2id(tag_path)
    f = open(word_path)
    lines = f.readlines()
    f.close()
    for line in lines:
        raw_str = line.split('\t')[0]
        tags = line.split('\t')[1]
        tag_list = tags.split(' ')
        word2tag_dict[raw_str] = tag_list
        word2tagid_dict[raw_str] = []
        for tag in tag_list:
            tagid = tag2id_dict[tag]
            word2tagid_dict[raw_str].append(tagid)
        for i in range(len(tag_list, VDS_LENGTH)):
            word2tag_dict[raw_str].append('NULL')
            word2tagid_dict[raw_str].append(0)
    return word2tag_dict, word2tagid_dict


# Function:
#    Add the VDS feature to each query
#      Each word may have multiple VDS tags, and these tags are separated by ' '
#      Different words are separated by '|'
# Input:
# Output:
def add_word_vds(data_path, tag2id_path):
    tag2id_dict, id2tag_list = read_tag2id(tag2id_path)
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f1 = codecs.open('data_vds_tag_' + data_set, 'w', encoding='utf-8')
    f2 = codecs.open('data_vds_id_' + data_set, 'w', encoding='utf-8')
    index = 0
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        words = sentence.split(' ')
        tag_dict = {}
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        qa_json = vc.get_anno_json(sentence, domain)
        # print qa_json
        try:
            anno = qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']
            for i in anno:
                if 'valid_data_type' in i['value']:
                    tag = i['value']['valid_data_type']
                    raw_str = i['raw_str']
                    if tag_dict.has_key(raw_str):
                        if tag not in tag_dict[raw_str]:
                            tag_dict[raw_str].append(tag)
                    else:
                        tag_dict[raw_str] = [tag, ]
            index += 1
            print sentence
        except Exception:
            print 'Error'

        tag_line = unicode(line[:-1] + '\t', 'utf-8')
        id_line = unicode(line[:-1] + '\t', 'utf-8')
        for word in words:
            word = unicode(word, 'utf-8')
            if tag_dict.has_key(word):
                tags = ''
                ids = ''
                for t in tag_dict[word]:
                    tid = str(tag2id_dict[t])
                    ids = ids + tid + ' '
                    tags = tags + t + ' '
                ids = ids[:-1]
                tags = tags[:-1]
            else:
                ids = '0'
                tags = 'NULL'

            tag_line = tag_line + tags + '|'
            id_line = id_line + ids + '|'
        tag_line = tag_line[:-1]+'\n'
        id_line = id_line[:-1]+'\n'
        f1.write(tag_line)
        f2.write(id_line)
    print index
    f1.close()
    f2.close()


# Function:
#    Add the VDS feature id (multihot) to each query
#      Each word may have multiple VDS tags, and these tags are separated by ' '
#      Different words are separated by '|'
# Input:
# Output:
def add_word_vds_multihot(data_path, tag2id_path):
    tag2id_dict, id2tag_list = read_tag2id(tag2id_path)
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = codecs.open('data_vds_id_multihot_' + data_set, 'w', encoding='utf-8')
    index = 0
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        words = sentence.split(' ')
        tag_dict = {}
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        qa_json = vc.get_anno_json(sentence, domain)
        try:
            anno = qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']
            for i in anno:
                if 'valid_data_type' in i['value']:
                    tag = i['value']['valid_data_type']
                    raw_str = i['raw_str']
                    if tag_dict.has_key(raw_str):
                        if tag not in tag_dict[raw_str]:
                            tag_dict[raw_str].append(tag)
                    else:
                        tag_dict[raw_str] = [tag, ]
            index += 1
            print sentence
        except Exception:
            print 'Error'
        tag_line = unicode(line[:-1] + '\t', 'utf-8')
        id_line = unicode(line[:-1] + '\t', 'utf-8')
        for word in words:
            word = unicode(word, 'utf-8')
            ids = ''
            id_multihot = np.zeros(VDS_SIZE)
            if tag_dict.has_key(word):
                for t in tag_dict[word]:
                    tid = tag2id_dict[t]
                    id_multihot[tid] = 1
            else:
                id_multihot[0] = 1
            for i in range(VDS_SIZE):
                 id_line = id_line + str(int(id_multihot[i])) + ' '
            id_line = id_line + '|'
        id_line = id_line[:-1]+'\n'
        f.write(id_line)
    print index
    f.close()


# Function:
#     Read the tag2id mapping file
# Input:
#     tag_path: path of the file that contains all the tags
# Output:
#      tag2id_dict: the dictionary that maps all tags to ids
#      id2tag_list: the list of all tags
def read_tag2id(tag_path):
    tag2id_dict = {'NULL': 0}
    id2tag_list = ['NULL', ]
    f = open(tag_path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        line = lines[i]
        tag = line.split(' ')[1]
        tag2id_dict[tag] = i+1
        id2tag_list.append(tag)
    # print tag2id_dict
    return tag2id_dict, id2tag_list


# Get vds feature online
def wget_vds(data_path, start, stop, version):
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = codecs.open('./data/data_vds_'+str(version)+'_' + data_set, 'w', encoding='utf-8')
    lines = lines[start:stop]
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = (sentence.replace(' ', ''))
        url = 'http://qa-alpha:5670/?query='+urllib.quote(sentence)+'&lang=zh-tw&debug=true&context=contextqa.default'
        print start, sentence
        qa_result = urllib.urlopen(url).read()
        qa_json = json.loads(qa_result)
        crf_feature_list = qa_json['debug']['Crf_debug_info']['Crf-feature_matrix-List']
        vd_line = unicode(line[:-1] + '\t', 'utf-8')
        for i in range(len(crf_feature_list)):
            features = crf_feature_list[i].split(' ')
            index = 3
            if features[3][-2:] != 'VD':
                for j in range(len(features)):
                    if features[j][-2:] == 'VD':
                        index = j
                        print 'VD is at index: %d' % index
                        break
            vd = features[index]
            vd_line = vd_line + vd + ' '
        vd_line = vd_line[:-1] + '\n'
        f.write(vd_line)
        start += 1
    f.close()


if __name__ == '__main__':
    data_set = 'train'
    data_path = './data/nlu.'+data_set+'.string.cnn_format'
    # analyse(data_path)  # get the tag list of data
    # split_data(data_path, 0.5)
    # get_taglist(data_path)
    # analyse(data_path)
    # get_combine('tag2id_list')
    # get_word2tag_map(data_path)
    # add_word_vds_multihot(data_path, '/home/qihu/PycharmProjects/cls_test/data/tag2id_list_all.txt')
    wget_vds(data_path, 40000, 40000, 3)
