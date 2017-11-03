#! /usr/bin/python

import numpy as np
from math import log
import re


# Calculate Mutual Information
def get_mi(pair_list, domain_num, feature_num):
    mutual_information = 0.0
    joint_table = np.zeros((domain_num, feature_num))
    domain_list = np.zeros(domain_num)
    feature_list = np.zeros(feature_num)
    n = len(pair_list)
    for pair in pair_list:
        joint_table[pair[0], pair[1]] += 1.0/n
        domain_list[pair[0]] += 1.0/n
        feature_list[pair[1]] += 1.0/n
    for i in range(domain_num):
        for j in range(feature_num):
            # print joint_table[i, j] / (domain_list[i] * feature_list[j])
            delta = joint_table[i, j]*log(joint_table[i, j]/(domain_list[i]*feature_list[j]+1e-10)+1e-10)/log(2)
            mutual_information += delta
    return mutual_information


# Read raw data from text file
def read_data(data_path):
    reg_pair_list = []
    vds_pair_list = []
    f = open(data_path)
    lines = f.readlines()
    f.close()
    for line in lines:
        label = line.split('\t')[0]
        regs = line.split('\t')[2].split(' ')
        vds = re.split(' |\|', line.split('\t')[3][:-1])
        for r in regs:
            reg_pair_list.append([label, r])
        for v in vds:
            if v != 'NULL':
                vds_pair_list.append([label, v])
    return reg_pair_list, vds_pair_list


# Read the text2id mapping dictionary
def read_label2id(label_path):
    label2id_dict = {}
    f = open(label_path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        label = lines[i][:-1]
        label2id_dict[label] = i
    return label2id_dict


# Convert the text-format pair_list to id-format
def convert2id(pair_list, domain_dict, feature_dict):
    id_pair_list = []
    for pair in pair_list:
        domain_id = domain_dict[pair[0]]
        feature_id = feature_dict[pair[1]]
        id_pair_list.append([domain_id, feature_id])
    return id_pair_list


# Convert the file to list-of-label format
#   Pre-process function (Not use in main process)
def convert_reg(file_path):
    f = open(file_path)
    lines = f.readlines()
    f.close()
    f = open('new_tmp.txt', 'w')
    for line in lines:
        label = line.split(' ')[1]
        f.write('%s\n' % label)
    f.close()


# Get the list-of-label format file from raw data
#   Pre-process function (Not use in main process)
def get_vds2id(data_path):
    f = open(data_path)
    lines = f.readlines()
    f.close()
    f = open('new_tmp.txt', 'w')
    vds_list = []
    for line in lines:
        vds = re.split(' ', line.strip('\n').split('\t')[2])
        for v in vds:
            if v not in vds_list:
                vds_list.append(v)
    for v in vds_list:
        f.write('%s\n' % v)
    f.close()


# main process
if __name__ == '__main__':
    data_path = 'data/nlu.regex_vds.all'
    domain_path = 'data/nlu.label_id.txt'
    reg_path = 'data/nlu.reg_id.txt'
    vds_path = 'data/nlu.vds_id.txt'

    # Read the  dictionaries mapping domain & RegEx & VDS to id
    domain2id_dict = read_label2id(domain_path)
    reg2id_dict = read_label2id(reg_path)
    vds2id_dict = read_label2id(vds_path)

    # The number of domain & RegEx & VDS classes
    domain_num = len(domain2id_dict)
    reg_num = len(reg2id_dict)
    vds_num = len(vds2id_dict)

    # Read raw data
    reg_list, vds_list = read_data(data_path)

    # Convert text-format data to id-format
    regid_list = convert2id(reg_list, domain2id_dict, reg2id_dict)
    vdsid_list = convert2id(vds_list, domain2id_dict, vds2id_dict)

    # Calcuate Mutual Information of domain-RegEx & domain-VDS
    reg_mi = get_mi(regid_list, domain_num, reg_num)
    vds_mi = get_mi(vdsid_list, domain_num, vds_num)

    print 'Mutual Information:\n\tRegEx: %.2f\n\tVDS: %.2f' % (reg_mi, vds_mi)
