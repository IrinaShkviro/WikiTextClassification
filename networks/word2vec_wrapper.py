# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:49:38 2017

@author: Admin
"""

import os
import time
import numpy as np
import gensim, logging
from pprint import pprint
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from doc_iterator import DocIterator
        
'''
class word2vec_wrapper is class for read docs from dir and create a word2vec 
model for this docs
we create word2vec model, create vocabulary on docs from "vocab_dir", 
then train it on docs from dir "train_dir"

use DocIterator for access to docs from dirs on disk

methods:
save_model()
    save the model on disk
tune_model()
    build the vocabulary, train the model and save it
get_word_list_by_id(doc_id) 
    return list of words from doc with id=doc_id       
get_next_doc()
    return pair(list_of_words_from_next_doc, id_next_doc)
get_doc_by_id(doc_id)
    return pair(list_of_words_from_doc, doc_id)
get_next_doc_id()
    return id of next doc
'''

class Word2VecWrapper(object):
    def __init__(self, vocab_dir, train_dir):
        self.vocab_docs = DocIterator(vocab_dir) # a memory-friendly iterator
        self.train_docs = DocIterator(train_dir) # a memory-friendly iterator
        self.save_dir_name = "saved"
        if not os.path.isdir(self.save_dir_name):
            os.mkdir(self.save_dir_name)
            
        self.dict_name = 'wiki_dict.dict'
        self.corpus_name = 'wiki_corpus.mm' 
        self.model_name = 'wiki_word2vec_model'

        # define word2vec parameters
    
        sentences = None
        self.dim = 200 # default is 100
        initial_learning_rate = 0.025
        maximum_distance = 5
        min_frequency = 5 # optional is between 0-100
        max_vocab_size = None
        random_downsample = 0.001
        min_learning_rate = 0.0001
        train_algo = 0 # 0 is CBOW and 1 is skip-gram
        hierarchical_softmax = 0 # 1 is hierarchical softmax will be used, 0 and negative is non-zero, negative sampling will be used
        negative = 5 # if > 0, negative sampling will be used, the int specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative samping is used.
        cbow_mean = 1 # 0 use the sum of the context word vectors; 1 use the mean (when cbow is used)
        n_epochs = 5 # number of iterations over the corpus
        sorted_vocab = 1 # 1 is sort the vocabulary by descending frequency before assigning word indexes
        batch_words = 10000  # target size (in words) for batches of examples passed to worker threads

        self.model = gensim.models.Word2Vec(
            sentences = sentences
            , size = self.dim
            , alpha = initial_learning_rate
            , window = maximum_distance
            , min_count = min_frequency
            , max_vocab_size = max_vocab_size
            , sample = random_downsample
            , min_alpha = min_learning_rate
            , sg = train_algo
            , hs = hierarchical_softmax
            , negative = negative
            , cbow_mean = cbow_mean
            , iter = n_epochs
            , sorted_vocab = sorted_vocab
            , batch_words = batch_words
        )
        
            
    def save_model(self):
        # save the model
        os.chdir(self.save_dir_name)
        self.model.save(self.model_name)
        os.chdir('..')

    def tune_model(self):
        # build the vocabulary
        self.model.build_vocab(self.vocab_docs)
        
        # how much word in the model
        # print(len(model.vocab))
        
        # train the model
        self.model.train(self.train_docs)  # can be a non-repeatable, 1-pass generator
        
        self.w2vdict = dict(zip(self.model.index2word, self.model.syn0))
    
        # less RAM (but can't change it!)
        # model.init_sims(replace=True)
        
        self.save_model()
        
    def get_word_list_by_id(self, doc_id):
        return self.train_docs.get_doc_by_id(doc_id)
        
    def get_next_doc(self):
        return self.train_docs.get_next_doc()
        
    def get_doc_by_id(self, doc_id):
        return self.train_docs.get_doc_by_id(doc_id)
        
    def get_next_doc_id(self):
        return self.train_docs.get_next_doc_id()
        



# print(model.vocab)
# print(model.estimate_memory())
# print(model.index2word)

# load model
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')


'''   
#####################
# create dictionary #
#####################

dictionary = corpora.Dictionary(docs)
os.chdir(dir_name)
dictionary.save(dict_name)  # store the dictionary, for future reference
os.chdir('..')
pprint(dictionary.token2id)


#################
# create corpus #
#################

# load dictionary
os.chdir(dir_name)
dictionary = corpora.Dictionary.load(dict_name)
os.chdir('..')  

class MyCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                # assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(line.lower().split())
             
corpus = [dictionary.doc2bow(doc) for doc in docs]

# save corpus
os.chdir(dir_name)
corpora.MmCorpus.serialize(corpus_name, corpus)  # store to disk, for later use
os.chdir('..')
pprint(corpus)


###########
#
###########

# load dictionary and corpus
os.chdir(dir_name)
dictionary = corpora.Dictionary.load(dict_name)
corpus = corpora.MmCorpus(corpus_name)
os.chdir('..')
'''