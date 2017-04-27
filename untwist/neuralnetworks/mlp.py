"""
Multi-Layer Perceptron implementation, using Theano
"""
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as T
try:
    import cPickle
except:
    import pickle as cPickle

from ..base import algorithms


class Layer(object):
    """
    Generic feed-forward layer (can be hidden or output)
    """

    def __init__(self, input, n_in, n_out,
                 W=None, b=None, activation=T.nnet.sigmoid):
        self.input = input
        if W is None:
            W_values = np.asarray(
                np.random.RandomState().uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(algorithms.Model):
    """
    Multi-Layer Perceptron network, specified by the size of the input,
    the output and an arbitrary number of hidden layers.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 activation=T.nnet.sigmoid):

        self.hidden_layers = []
        self.params = []
        self.input = T.matrix('x')
        self.target = T.matrix('y')

        for i, layer_size in enumerate(hidden_sizes):
            if i == 0:
                layer_input_size = input_size
                layer_input = self.input
            else:
                layer_input_size = hidden_sizes[i - 1]
                layer_input = self.hidden_layers[-1].output
            layer = Layer(
                layer_input,
                layer_input_size,
                layer_size,
                activation=activation)
            self.hidden_layers.append(layer)
            self.params.extend(layer.params)

        self.output_layer = Layer(
            self.hidden_layers[-1].output, hidden_sizes[-1],
            output_size)
        self.params.extend(self.output_layer.params)
        self.output = self.output_layer.output
        self.cost = T.sum((self.output - self.target)**2)

    def save(self, fname):
        hidden = [{'W': layer.W.eval(), 'b': layer.b.eval()}
                  for layer in self.hidden_layers]
        output = {
            'W': self.output_layer.W.eval(),
            'b': self.output_layer.b.eval()}
        model_params = {'hidden': hidden, 'output': output}
        with open(fname, 'wb') as f:
            cPickle.dump(model_params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            new_params = cPickle.load(f)
        for i, l in enumerate(self.hidden_layers):
            l.W.set_value(new_params['hidden'][i]['W'])
            l.b.set_value(new_params['hidden'][i]['b'])
        self.output_layer.W.set_value(new_params['output']['W'])
        self.output_layer.b.set_value(new_params['output']['b'])


class Activations:
    """
    Some activation functions not available in theano.
    """

    @classmethod
    def ReLU(cls):
        return lambda x: T.switch(x < 0, 0, x)

    @classmethod
    def ReLU2(cls):
        return lambda x: T.log(1+T.exp(x))
