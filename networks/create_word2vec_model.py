# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:11:35 2017

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:30:39 2017

@author: Admin
"""

import os
import logging
import gensim
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from doc_iterator import DocIterator

def create_word2vec(vocab_dir = 'preproceed', train_dir = 'preproceed'):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    vocab_dir = os.path.join(this_folder, '../' + vocab_dir)
    train_dir = os.path.join(this_folder, '../' + train_dir)
    saved_dir = os.path.join(this_folder, '../saved')
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)
     
    
    # define word2vec parameters    
    sentences = None
    dim = 200 # default is 100
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

    w2v_model = gensim.models.Word2Vec(
        sentences = sentences
        , size = dim
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
    
    vocab_docs = DocIterator(vocab_dir)
    
    train_docs = DocIterator(train_dir)
    
    
    # build the vocabulary
    w2v_model.build_vocab(vocab_docs)
        
    # train the model
    train_samples = train_docs.calc_total_words()
    w2v_model.train(train_docs, total_examples = train_samples, epochs = w2v_model.iter)
    
    w2vdict = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
    
    #save the model    
    os.chdir(saved_dir)
    w2v_model.save('w2v_model')
    file = open('w2v_dict', 'wb')
    pickle.dump(w2vdict, file)
    file.close()
    os.chdir('../networks')
           
    # less RAM (but can't change it!)
    # model.init_sims(replace=True)
   
    
if __name__ == '__main__':
    create_word2vec()