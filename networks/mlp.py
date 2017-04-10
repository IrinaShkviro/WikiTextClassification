# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:21:48 2017

@author: Admin
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data
from calc_n_labels import calc_n_labels

"""
class MLP is a logistic regressor, where instead of feeding the input to 
logistic regression we insert an intermediate layer, named HiddenLayer, that
has a nonlinear activation function. 
    class HiddenLayer is a class that works as logistic regression 
    (output = W * input + b), but have some nonlinear activation function
    (output = activation(w * input + b)). This function usually tahn or sigmoid
MLP = HiddenLayer + LogisticRegression
So input comes first at hidden layer, than output of hidden layer comes as
input of logistic regression classifier and output of classifier is output of
MLP

class methods:
negative_log_likelihood(y)
    method from Logisticregression class
    return the mean of the negative log-likelihood of the prediction of this 
    model under a given target distribution  
errors(y):
    method from Logisticregression class
    return a float representing the number of errors in the minibatch over 
    the total number of examples of the minibatch ; zero one loss over 
    the size of the minibatch

external methods:
test_mlp(learning_rate, n_epochs, batch_size)
    load train, valid and test sets
    create class LogisticRegression and train it for n_epochs
    save the best trained model
    
"""

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
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

    # compute number of minibatches for training, validating and testing
    (n_train_batches, n_last_train_batch) = divmod(train_vectorizer.calc_n_samples(), batch_size)
    if (n_last_train_batch > 0):
        n_train_batches = n_train_batches + 1
    (n_valid_batches, n_last_valid_batch) = divmod(valid_vectorizer.calc_n_samples(), batch_size) 
    if (n_last_valid_batch > 0):
        n_valid_batches = n_valid_batches + 1
    (n_test_batches, n_last_test_batch) = divmod(test_vectorizer.calc_n_samples(), batch_size)   
    if (n_last_test_batch > 0):
        n_test_batches = n_test_batches + 1

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    y_list = T.imatrix('y_list')  # labels list, presented as vectors

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng = rng,
        input = x,
        n_in = train_vectorizer.dim,
        n_hidden = n_hidden,
        n_out = n_labels
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
 
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs = [x, y_list],
        outputs = classifier.errors(y_list)
    )

    validate_model = theano.function(
        inputs = [x, y_list],
        outputs = classifier.errors(y_list)
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
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
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    train_arr = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for batch_id in range(n_train_batches):
            
            next_train_batch = train_vectorizer.get_next_batch(batch_size = batch_size)
            train_features = next_train_batch[0].get_value(
                borrow = True,
                return_internal_type = True
            )
            train_labels = next_train_batch[1].eval()
            batch_avg_cost = train_model(train_features, train_labels)

            # iteration number
            iter = (epoch - 1) * n_train_batches + batch_id
            train_arr.append((iter, batch_avg_cost))

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
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

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

                    print(('     epoch %i, batch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, batch_id + 1, n_train_batches,
                           test_score * 100.))
                           
                     # save the best model
                    with open('best_model_mlp.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_mlp(n_epochs = 5)