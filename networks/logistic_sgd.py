# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:00:29 2017

@author: Admin
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from text_vectorizer import TextVectorizer
from calc_n_labels import calc_n_labels

"""
class LogisticRegression is probabilistic, linear classifier. This class use
stochastic gradient descent optimization method. Classification is done by 
projecting data points onto a set of hyperplanes, the distance to which is 
used to determine a class membership probability.
The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).


class methods:
negative_log_likelihood(y)
    return the mean of the negative log-likelihood of the prediction of this 
    model under a given target distribution  
errors(y):
    return a float representing the number of errors in the minibatch over 
    the total number of examples of the minibatch ; zero one loss over 
    the size of the minibatch

external methods:
load_data(vocab_dir, train_dir, vectorizer)
    return object of class TextVectorizer for reading a train data
sgd_optimization(learning_rate, n_epochs, batch_size)
    load train, valid and test sets
    create claa LogisticRegression and train it for n_epochs
    save the best trained model
predict()
    load the best trained model
    for each given x return y that most suitable
"""

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
                
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
    def errors(self, y_lists):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y_lists: theano.tensor.TensorType
        :param y_lists: corresponds to a list of vectors that gives for each example the
                  correct label
        """
            
        return T.mean(1 - T.any(T.switch(T.eq(self.y_pred, T.transpose(y_lists)), 
                                         T.ones_like(self.y_pred),
                                        T.zeros_like(self.y_pred)), axis = 0))

def load_data(train_dir = 'preproceed', vectorizer = MeanEmbeddingVectorizer):
    ''' Loads the dataset

    :type vocab_dir: string
    :param vocab_dir: the name of dir with vocabulary 
    
    :type train_dir: string
    :param train_dir: the name of dir with train data 
    
    :type vectorizer: object
    :param vectorizer: the path to the dataset 
    '''

    #############
    # LOAD DATA #
    #############
    
    text_vectorizer = TextVectorizer (
        vectorizer = vectorizer,
        train_dir = train_dir
    )
    
    return text_vectorizer
    
def save_model(model, filename, dir_name = 'saved'):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    file = open(filename, 'wb')
    pickle.dump(model, file)
    file.close()
    os.chdir('..')

def load_model(filename, dir_name = 'saved'):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    saved_folder = os.path.join(this_folder, 'saved')
    os.chdir(saved_folder)
    
    file = open(filename, 'rb')
    # load the saved model
    model = pickle.load(file)
    file.close()
    os.chdir('..')
    return model

def sgd_optimization(learning_rate=0.13, n_epochs=1000,
                           batch_size=6):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    """
    
    n_labels = calc_n_labels()
    print(n_labels, 'n labels')
    
    train_vectorizer = load_data(
        train_dir = 'train_set'
    )
    
    test_vectorizer = load_data(
        train_dir = 'test_set'
    )

    # compute number of batches for training and testing
    (n_train_batches, n_last_train_batch) = divmod(train_vectorizer.calc_n_samples(), batch_size)
    if (n_last_train_batch > 0):
        n_train_batches = n_train_batches + 1
    (n_test_batches, n_last_test_batch) = divmod(test_vectorizer.calc_n_samples(), batch_size)   
    if (n_last_test_batch > 0):
        n_test_batches = n_test_batches + 1
 
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as vectors
    y = T.ivector('y')  # labels, presented as int labels
    y_list = T.imatrix('y_list')  # labels list, presented as vectors

    # construct the logistic regression class
    classifier = LogisticRegression(
        input = x,
        n_in = train_vectorizer.dim,
        n_out = n_labels
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs = [x, y_list],
        outputs = classifier.errors(y_list)
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=updates
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    test_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_test_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    best_iter = 0
    train_arr = []
    
    file_name = "log_reg_log.txt"
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        sys.stdout.write("%i / %i \n" % (epoch, n_epochs))
        for batch_id in range(n_train_batches):

            next_train_batch = train_vectorizer.get_next_batch(batch_size = batch_size)
            train_features = next_train_batch[0].get_value(
                borrow=True,
                return_internal_type=True
            )
            train_labels = next_train_batch[1].eval()
            batch_avg_cost = train_model(train_features, train_labels)
            
            f = open(file_name, 'w')
            f.write(str(batch_avg_cost) + '\n')
            f.close()
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + batch_id
            train_arr.append((iter, batch_avg_cost))

            if (iter + 1) % test_frequency == 0:
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

                print(
                    'epoch %i, batch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        batch_id + 1,
                        n_train_batches,
                        this_test_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_test_loss < best_test_loss:
                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_test_loss = this_test_loss
                    best_iter = iter
                    
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            batch_id + 1,
                            n_train_batches,
                            this_test_loss * 100.
                        )
                    )

                    # save the best model
                    save_model(model = classifier, filename = 'best_model_log_reg.pkl')

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%'
        )
        % (best_test_loss * 100., best_iter + 1, test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = load_model(filename = 'best_model_log_reg.pkl')

    # compile a predictor function
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = classifier.y_pred
    )

    # We can test it on some examples from test test
    '''
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    '''


if __name__ == '__main__':
    sgd_optimization(n_epochs = 5)