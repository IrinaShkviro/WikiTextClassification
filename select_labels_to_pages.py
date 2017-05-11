# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:27:14 2017

@author: Admin
"""

import pymysql as mysql
import numpy as np
import os
import pickle
import sys

from pprint import pprint
#import mysql.connector
'''
cnx = mysql.connector.connect(user='root', password='DbMysqlP@ss2016',
                              host='195.19.233.35',port=33061,
                              database='wiki')
'''
cnx = mysql.connect(user='root', password='DbMysqlP@ss2016',
                              host='localhost',port=3306,
                              database='wiki')
                              
curs = cnx.cursor()

def select_labels_by_id(cur_id):
    curs.execute('select cl_to from categorylinks \
        where cl_from = "cur_page" and cl_type = "page"'.replace('cur_page', cur_id))
    return curs.fetchall()
    
def decode_from_bytearray(sample):
    return sample[0].decode("utf-8") 

def select_labels_by_id_list(id_list):
    sys.stdout.write("select labels by id list... \n")
    data = []
    file_num = 0
    dir_size = len(id_list)
    for cur_id in id_list:
        file_num = file_num + 1
        labels = select_labels_by_id(cur_id)
        cat_list = [decode_from_bytearray(label) for label in labels]
        data.append((cur_id, cat_list))
        sys.stdout.write("%i / %i \r" % (file_num, dir_size))
    sys.stdout.write("select %i labels \n" % dir_size)
    return dict(data)
    
def select_id_list_from_dir(dir_name):
    sys.stdout.write("select id list from dir... \n")
    dir_size =  (len(os.listdir(dir_name)))
    id_list = []
    file_num = 0
    for fname in os.listdir(dir_name):
        file_num = file_num + 1
        end = fname.find('.')
        id_list.append(fname[3:end])
        sys.stdout.write("%i / %i \r" % (file_num, dir_size))
    sys.stdout.write("find %i ids \n" % len(id_list))
    return id_list

def create_dict_for_categories(train_dict):
    sys.stdout.write("create dict for categories... \n")
    dict_size = 0
    cat_codes = dict()
    n_labels = len(train_dict.values())
    label_num = 0
    for label_list in train_dict.values():
        label_num = label_num + 1
        for label in label_list:
            if not label in cat_codes:
                cat_codes[label] = dict_size
                dict_size = dict_size + 1
        sys.stdout.write("%i / %i \r" % (label_num, n_labels))
    sys.stdout.write("create dictionary with %i different categories \n" % len(cat_codes))
    return cat_codes
    
def prepare_train_set(dir_name):
    sys.stdout.write("prepare the train set...\n")
    page_id_list = select_id_list_from_dir(dir_name)
    train_dict = select_labels_by_id_list(page_id_list)
    cat_codes = create_dict_for_categories(train_dict)
    sys.stdout.write("parepare numeric train set...\n")
    train_data = []
    n_train_samples = len(train_dict.items())
    cur_sample = 0
    for x, labels in train_dict.items():
        cur_sample = cur_sample + 1
        label_ids = []
        for label in labels:
            label_ids.append(cat_codes.get(label))
        train_data.append((x, label_ids))
        sys.stdout.write("%i / %i \r" % (cur_sample, n_train_samples))
    return dict(train_data), cat_codes
    
def save_dict(cur_dict, filename, dir_name = 'saved'):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    file = open(filename, 'wb')
    pickle.dump(cur_dict, file)
    file.close()
    os.chdir('..')
    
# train_dict = {(page_id, [categories_id])}
# cat_dict = {(category_name, category_id)}
train_dict, cat_dict = prepare_train_set('preproceed')

save_dict(train_dict, 'train_data')
save_dict(cat_dict, 'category_codes')

# test_train = open('train_data', 'rb')
# data_new = pickle.load(test_train)

print(len(train_dict), "train count")
print(len(cat_dict), "categories count")

cnx.close()