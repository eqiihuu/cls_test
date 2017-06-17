# coding=utf-8

from utils import vds_caller as vc


def analyse(data_path):
    f = open(data_path)
    lines = f.readlines()
    i = 0
    for line in lines:
        domain = line.split('\t')[0]
        sentence = line.split('\t')[1]
        sentence = unicode(sentence.replace(' ', ''), 'utf-8')
        print i, sentence, domain
        i += 1
        vc.get_annotations(sentence, domain)

def merge(src_path, dest_path):
    f_src = open(src_path)
    content = f_src.readlines()
    f_src.close()
    f_dest = open(dest_path, 'a')
    f_dest.write(content)
    f_dest.close()
    print 'Merge %s to %s' %(src_path, dest_path)

if __name__ == '__main__':
    data_path = './cnn/data/nlu.train.string.cnn_format'
    analyse(data_path)

