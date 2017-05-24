# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:53:22 2017

@author: Admin
"""

import os
import sys
import timeit

import theano
import theano.tensor as T

import numpy

from logistic_sgd import load_data, load_model, LogisticRegression

def test_valid(model_name, file_name, batch_size):
    
    start_time = timeit.default_timer()
    
    test_vectorizer = load_data(
        train_dir = 'test_set'
    )
    
    (n_test_batches, n_last_test_batch) = divmod(test_vectorizer.calc_n_samples(), batch_size)   
    if (n_last_test_batch > 0):
        n_test_batches = n_test_batches + 1
            
    # load the saved model
    classifier = load_model(model_name)
    
    y_list = T.imatrix('y_list')  # labels list, presented as vectors
    
    test_model = theano.function(
        inputs = [classifier.input, y_list],
        outputs = classifier.errors(y_list),
        on_unused_input = 'ignore'
    )
    
    # compute zero-one loss on validation set
    test_losses = []
    for cur_test_batch in range(n_test_batches):
        next_test_batch = test_vectorizer.get_next_batch(batch_size = batch_size)
        test_features = next_test_batch[0].get_value(
            borrow=True,
            return_internal_type=True
        )
        labels_list = next_test_batch[2].eval()
        test_losses.append(test_model(test_features, labels_list))
                    
    this_test_loss = numpy.mean(test_losses)
    
    end_time = timeit.default_timer()
    
    f = open(file_name, 'w')
    f.write('validation error %f %% \n' % (this_test_loss * 100.))
    f.write(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
    f.close()
    

if __name__ == '__main__':
    model_name = sys.argv[1]
    file_name = sys.argv[2]
    batch_size = sys.argv[3]
    test_valid(model_name, file_name, int(batch_size))