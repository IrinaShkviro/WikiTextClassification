# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:33:20 2017

@author: Admin
"""

import numpy as np

'''
class MeanEmbeddingVectorizer is a class for representing one document as vector
this class has a word2vec model as parameter and make transformations with it help
for each doc:
    1. transforn each word of this doc to vector
    2. calc mean vector by all vector words in doc

methods:
transform(word_list) 
    return one vector, representing this word_list
'''


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec.w2vdict
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.dim

    def fit(self, X, y):
        return self

    def transform(self, word_list):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in word_list
        ])
        
    def transform_one_list(self, word_list):
        return np.mean([self.word2vec[word] for word in word_list if word in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
