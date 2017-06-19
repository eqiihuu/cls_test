# coding=utf-8

import random
import json
from utils import vds_caller as vc


def analyse(data_path):
    f = open(data_path)
    lines = f.readlines()
    i = 0
    for line in lines[0:2]:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        print i, domain, sentence
        i += 1
        anno = vc.get_annotations(sentence, domain)
    f.close()


def merge(src_path, dest_path):
    f_src = open(src_path)
    content = f_src.readlines()
    f_src.close()
    f_dest = open(dest_path, 'a')
    f_dest.write(content)
    f_dest.close()
    print 'Merge %s to %s' % (src_path, dest_path)


# Split a file of data into two part, you can specify the percentage
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


# Get the word list of a sentence
def get_wordlist():
    wordlist = []
    sentence = u"沭阳到苏州的车票"
    domain = "nlu.navigation"
    anno = vc.get_annotations(sentence, domain)
    print len(anno)
    for i in anno:
        print i.tag, i.raw_str
    return wordlist

if __name__ == '__main__':
    data_path = '/home/qihu/PycharmProjects/cls_test/data/nlu.dev.string.cnn_format'
    # analyse(data_path)  # get the word list of data
    # split_data(data_path, 0.5)
    get_wordlist()
    # analyse(data_path)


