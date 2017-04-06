# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:54:29 2017

@author: Admin
"""

import os
import numpy as np

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
        self.dir_name = dir_name
        self.doc_list = os.listdir(self.dir_name)
        self.n_files = len(self.doc_list)
        self.cur_doc_num = 0
        # remove common words and tokenize
        self.stoplist = set('for a of the and to in'.split())
        self.punctuation = set(', . : ; ! ? " ( ) [ ] { } # & ^ \ | / $'.split()) 
        
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
        for line in open(os.path.join(self.dir_name, fname)):
            word_list = [word for word in line.lower().split() if word not in self.stoplist]
                
            for word_num in np.arange(len(word_list), dtype = int):
                word = word_list[word_num]
                for punct in self.punctuation:
                    word = word.replace(punct, "")
                word_list[word_num] = word
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