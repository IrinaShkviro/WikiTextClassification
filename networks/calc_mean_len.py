# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:57:55 2017

@author: Admin
"""

import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from doc_iterator import DocIterator
from text_vectorizer import TextVectorizer

import numpy as np
import sys

from logistic_sgd import load_data

def calc_mean_len(dir_name = 'preproceed'):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    #dir_name = os.path.join(this_folder, '../' + dir_name)
    saved_dir = os.path.join(this_folder, '../saved')
        
    train_vectorizer = load_data(
        train_dir = dir_name
    )
    
    lens = []
    for i in range(train_vectorizer.n_files):
        next_doc = train_vectorizer.get_next_doc_vector()
        cur_page_id = next_doc[1]
        cur_label_list = train_vectorizer.train_dict.get(cur_page_id)
        lens.append((len(cur_label_list)))
        
        sys.stdout.write("%i / %i \r" % (i, train_vectorizer.n_files))
        
    mean_res = np.mean(np.asarray(lens))
    print('mean labels: ', mean_res)
       

if __name__ == '__main__':
    calc_mean_len(sys.argv[1])