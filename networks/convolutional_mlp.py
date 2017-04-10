# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:57:07 2017

@author: Admin


This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

"""

from __future__ import print_function

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from calc_n_labels import calc_n_labels
from networks.mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, data_shape, poolsize = (1, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic data tensor, of shape data_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type data_shape: tuple or list of length 4
        :param data_shape: (batch size, num input feature maps,
                             data height, data width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert data_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            input_shape = data_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input = conv_out,
            ws = poolsize,
            ignore_border = True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class CNN_4_levels(object):
     def __init__(self, layer0, layer1, layer2, layer3):
         
         # layer0 is LeNetConvPoolLayer
         self.layer0 = layer0
         
         # layer1 is LeNetConvPoolLayer
         self.layer1 = layer1
         
         # layer2 is HiddenLayer from mlp
         self.layer2 = layer2
         
         # layer3 is classifier (logistic regression)
         self.layer3 = layer3
        
         

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    n_labels = calc_n_labels()
    print(n_labels, 'n labels')
    
    train_vectorizer = load_data(
        vocab_dir = 'preproceed',
        train_dir = 'preproceed'
    )
    
    valid_vectorizer = load_data(
        vocab_dir = 'preproceed',
        train_dir = 'preproceed'
    )
    
    test_vectorizer = load_data(
        vocab_dir = 'preproceed',
        train_dir = 'preproceed'
    )

    # compute number of batches for training, validating and testing
    (n_train_batches, n_last_train_batch) = divmod(train_vectorizer.calc_n_samples(), batch_size)
    if (n_last_train_batch > 0):
        n_train_batches = n_train_batches + 1
    (n_valid_batches, n_last_valid_batch) = divmod(valid_vectorizer.calc_n_samples(), batch_size) 
    if (n_last_valid_batch > 0):
        n_valid_batches = n_valid_batches + 1
    (n_test_batches, n_last_test_batch) = divmod(test_vectorizer.calc_n_samples(), batch_size)   
    if (n_last_test_batch > 0):
        n_test_batches = n_test_batches + 1
 
    x = T.matrix('x')   # the data is presented as vectors
    y = T.ivector('y')  # the labels are presented as int labels
    y_list = T.imatrix('y_list')  # labels list, presented as vectors

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    # n_in0 = 200
    n_in0 = train_vectorizer.dim

    # Reshape matrix of rasterized images of shape (batch_size, dim)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # dim is the size of one input vector
    layer0_input = x.reshape((batch_size, 1, 1, n_in0))

    # Construct the first convolutional pooling layer:
    # filtering reduces the data size to (1 - 1 + 1 , n_in0 - 5 + 1) = (1, n_in0 - 4) = (1, 196)
    # maxpooling reduces this further to (1 / 1, (n_in0 - 4) / 4) = (1, n_in0 / 4 - 1) = (1, 49)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 1, n_in0 / 4 - 1) = (batch_size, nkerns[0], 1, 49)
    layer0 = LeNetConvPoolLayer(
        rng,
        input = layer0_input,
        data_shape = (batch_size, 1, 1, n_in0),
        filter_shape = (nkerns[0], 1, 1, 5),
        poolsize = (1, 4)
    )

    # n_in1 = 49
    n_in1 = n_in0 // 4 - 1
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (1 - 1 + 1, n_in1 - 5 + 1) = (1, n_in1 - 4) = (1, 45)
    # maxpooling reduces this further to (1 / 1, (n_in1 - 4) / 3) = (1, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 1, (n_in1 - 4) / 3) = (batch_size, nkerns[1], 1, 15)
    layer1 = LeNetConvPoolLayer(
        rng,
        input = layer0.output,
        data_shape = (batch_size, nkerns[0], 1, n_in1),
        filter_shape = (nkerns[1], nkerns[0], 1, 5),
        poolsize = (1, 3)
    )

    # n_in_hidden = 15
    n_in_hidden = (n_in1 - 4) // 3
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_elems)
    # This will generate a matrix of shape (batch_size, nkerns[1] * 1 * 15),
    # or (500, 50 * 1 * 15) = (500, 750) with the default values
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input = layer2_input,
        n_in = nkerns[1] * n_in_hidden,
        n_out = batch_size,
        activation = T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input = layer2.output, n_in = batch_size, n_out = n_labels)
    
    cnn_4 = CNN_4_levels(layer0, layer1, layer2, layer3)

    # the cost we minimize during training is the NLL of the model
    cost = cnn_4.layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x, y_list],
        cnn_4.layer3.errors(y_list)
    )

    validate_model = theano.function(
        [x, y_list],
        cnn_4.layer3.errors(y_list)
    )

    # create a list of all model parameters to be fit by gradient descent
    params = cnn_4.layer3.params + cnn_4.layer2.params + cnn_4.layer1.params + cnn_4.layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x, y],
        cost,
        updates=updates
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    train_arr = []
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for batch_id in range(n_train_batches):
            
            next_train_batch = train_vectorizer.get_next_batch(batch_size = batch_size)
            train_features = next_train_batch[0].get_value(
                borrow=True,
                return_internal_type=True
            )
            train_labels = next_train_batch[1].eval()
            batch_avg_cost = train_model(train_features, train_labels)

            iter = (epoch - 1) * n_train_batches + batch_id
            train_arr.append((iter, batch_avg_cost))

            if iter % 100 == 0:
                print('training iter = ', iter)      

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                for cur_valid_batch in range(n_valid_batches):
                    next_valid_batch = valid_vectorizer.get_next_batch(batch_size = batch_size)
                    valid_features = next_valid_batch[0].get_value(
                        borrow=True,
                        return_internal_type=True
                    )
                    labels_list = next_train_batch[2].eval()
                    validation_losses.append(validate_model(valid_features, labels_list))
                    
                this_validation_loss = numpy.mean(validation_losses)
                
                print(
                    'epoch %i, batch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        batch_id + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = []
                    for cur_test_batch in range(n_test_batches):
                        next_test_batch = test_vectorizer.get_next_batch(batch_size = batch_size)
                        test_features = next_test_batch[0].get_value(
                            borrow=True,
                            return_internal_type=True
                        )
                        labels_list = next_train_batch[2].eval()
                        test_losses.append(test_model(test_features, labels_list))

                    test_score = numpy.mean(test_losses)
                    
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            batch_id + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    
                    # save the best model
                    with open('best_model_cnn_4.pkl', 'wb') as f:
                        pickle.dump(cnn_4, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    evaluate_lenet5(n_epochs = 5, batch_size = 20)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)