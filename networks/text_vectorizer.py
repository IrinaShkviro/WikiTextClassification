# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:32:19 2017

@author: Admin
"""
import numpy as np
import os
import pickle
import gc

import theano
import theano.tensor as T

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

from word2vec_wrapper import Word2VecWrapper
os.chdir('..')

'''
class text_vectorizer is class for transformation from docs to vectors 

use one of vectorizers to transform list of words to one vector
use Word2VecWrapper for access to docs from dirs on disk

methods:
calc_n_samples()
    return count of pair (x, y) in train set, where
        x - vector of document
        y - one label of document
        
get_next_pair()
    return pair (x, y) from train set, where
        x - vector of document
        y - one label of document 

get_next_batch(batch_size)
    return batch_size of train pairs
    
get_next_doc_vector()
    return pair (x, y), where
        x - one vector that characterize next doc
        y - next doc id
        
get_next_vector_batch(batch_size)
    return matrix, where each row characterize one next doc and 
    each row is pair (x, y), where       
        x - one vector that characterize doc
        y - doc's id
        
get_doc_vector_by_id(doc_id)
    return one vector that characterize doc with id=doc_id  
'''

class TextVectorizer(object):
    def __init__(self, vectorizer, vocab_dir = 'preproceed', train_dir = 'preproceed'):
        self.w2vmodel = Word2VecWrapper(
            vocab_dir = vocab_dir,
            train_dir = train_dir
        )
        self.w2vmodel.tune_model()
        self.vectorizer = vectorizer(self.w2vmodel)
        self.n_files = self.w2vmodel.train_docs.n_files
        self.dim = self.w2vmodel.dim
        
        os.chdir('saved')
        train_file = open('train_data', 'rb')
        self.train_dict = pickle.load(train_file)
        os.chdir('..')
        
        self.cur_x = []
        self.cur_label_list = []
        self.cur_label_num = 0
        self.n_cur_labels = 0
        
    def calc_n_samples(self):
        n_samples = 0
        n_diff_labels = set()
        for cur_doc in range(self.n_files):
            cur_doc_id = self.w2vmodel.get_next_doc_id()
            cur_label_list = self.train_dict.get(cur_doc_id)
            n_cur_labels = len(cur_label_list)
            n_samples = n_samples + n_cur_labels
        return n_samples            
        
    def get_next_pair(self):
        self.cur_label_num = self.cur_label_num + 1
        if self.cur_label_num <= self.n_cur_labels:
            return (self.cur_x, self.cur_label_list[self.cur_label_num - 1])
        else:
            next_doc = self.get_next_doc_vector()
            self.cur_x = next_doc[0]
            cur_page_id = next_doc[1]
            self.cur_label_list = self.train_dict.get(cur_page_id)
            self.n_cur_labels = len(self.cur_label_list)
            self.cur_label_num = 0
            return self.get_next_pair()
            
    def get_next_batch(self, batch_size):
        # x.shape = (200, )
        (cur_x, cur_y) = self.get_next_pair()
        x_arr = cur_x
        y_arr = cur_y
        for sample_num in range(batch_size - 1):
            (cur_x, cur_y) = self.get_next_pair()
            x_arr = np.vstack((x_arr, cur_x))
            y_arr = np.append(y_arr, cur_y)
            gc.collect()
        
        set_features = theano.shared(np.asarray(x_arr, dtype=theano.config.floatX),
                                     borrow=True)
        set_labels = T.cast(theano.shared(np.asarray(y_arr, dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        return (set_features, set_labels)
        
    def get_next_doc_vector(self):
        next_doc = self.w2vmodel.get_next_doc()
        return (self.vectorizer.transform_one_list(next_doc[0]), next_doc[1])
        
    def get_next_vector_batch(self, batch_size):
        arr = self.get_next_doc_vector()
        for cur_sample in range(batch_size - 1):
            arr = np.row_stack((arr, self.get_next_doc_vector()))
        return arr
        
    def get_doc_vector_by_id(self, doc_id):
        return self.vectorizer.transform_one_list(self.w2vmodel.get_word_list_by_id(doc_id))