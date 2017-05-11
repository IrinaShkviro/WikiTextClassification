# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:54:29 2017

@author: Admin
"""

import os
import sys
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

'''
class DocIterator is iterator class for access to docs from dirs on disk
it also remove punctuation and some frequent words from docs

methods:
get_word_list_by_id(doc_id) 
    return list of words from doc with id=doc_id       
get_next_doc()
    return pair(list_of_words_from_next_doc, id_next_doc)
get_doc_by_id(doc_id)
    return pair(list_of_words_from_doc, doc_id)
get_next_doc_id()
    return id of next doc
'''

class DocIterator(object):
    def __init__(self, dir_name):
        self.doc_folder = dir_name
        self.doc_list = os.listdir(self.doc_folder)
        self.n_files = len(self.doc_list)
        self.cur_doc_num = 0
        self.lemmatizer = WordNetLemmatizer()
        # remove common words and tokenize
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stoplist = stopwords.words('english')
        #self.punctuation = set(', . : ; ! ? " ( ) [ ] { } # & ^ \ | / $'.split()) 
        
    def get_next_doc_id(self):
        fname = self.doc_list[self.cur_doc_num]
        self.cur_doc_num = self.cur_doc_num + 1
        if self.cur_doc_num >= self.n_files:
            self.cur_doc_num = 0
        end = fname.find('.')
        doc_id = fname[3:end]
        return doc_id
        
    def get_word_list_by_name(self, fname):
        doc_word_list = []
        for line in open(os.path.join(self.doc_folder, fname)):
            tokens = self.tokenizer.tokenize(line.lower())
            word_list = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stoplist]
            doc_word_list.extend(word_list)
                        
        return doc_word_list
 
    def __iter__(self):
        for fname in self.doc_list:
            #print(fname)
            yield self.get_word_list_by_name(fname)
            
    def get_doc_by_id(self, doc_id):
        fname = "id_" + doc_id + ".txt"
        return (self.get_word_list_by_name(fname), doc_id)
        
    def get_next_doc(self):
        fname = self.doc_list[self.cur_doc_num]
        self.cur_doc_num = self.cur_doc_num + 1
        if self.cur_doc_num >= self.n_files:
            self.cur_doc_num = 0
        end = fname.find('.')
        doc_id = fname[3:end]
        return (self.get_word_list_by_name(fname), doc_id)
        
    def calc_total_words(self):
        total_words = 0
        for cur_doc in range(self.n_files):
            total_words = total_words + len(self.get_next_doc()[0]) 
            sys.stdout.write("%i / %i \r" % (cur_doc, self.n_files))
        print('total words: ', total_words)
        return total_words