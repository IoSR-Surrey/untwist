import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

class SGD(object):

    def __init__(self, mlp, learning_rate = 0.1, 
        momentum = 0.5, batch_size = 100, iterations = 100):
        self.mlp = mlp
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.training_error = []
        cost = mlp.cost /(2 * batch_size)
        self.gparams = [T.grad(cost, param) for param in mlp.params]
        self.param_updates = [
            theano.shared(value = np.zeros(
                param.get_value().shape, 
                dtype = theano.config.floatX)
            )
            for param in mlp.params
        ]
        self.updates = []
        for param, gparam, uparam in zip(
            mlp.params, 
            self.gparams, 
            self.param_updates):
                
            self.updates.append(
                (param, param - learning_rate * uparam)
            )
            
            self.updates.append(
                (uparam, (momentum * uparam) + ((1. - momentum) * gparam))
            )
         
        n_bins = mlp.hidden_layers[0].W.shape.eval()[0]

        self.xi = theano.shared( 
            name  = 'xi',  
            value = np.zeros((self.batch_size,n_bins), dtype = floatX),
            allow_downcast = True)
            
        self.yi = theano.shared(
            name  = 'yi', 
            value = np.zeros((self.batch_size,n_bins), dtype = floatX),
            allow_downcast = True)

        self.train_func = theano.function(
            inputs = [],
            outputs = cost,
            updates = self.updates,
            allow_input_downcast = True,
            givens = {mlp.input:self.xi, mlp.target:self.yi}
        )
        
        self.predict_func = theano.function(
            inputs = [],
            outputs = mlp.output,
            givens = {mlp.input:self.xi},
            allow_input_downcast = True
        )


    def train(self, dataset):
        n_batches = dataset.X.shape[0] / self.batch_size

        for epoch in range(self.iterations):
            batch_cost = 0
            for index in xrange(n_batches):
                batch = dataset.get_batch(index, self.batch_size)      
                self.xi.set_value(np.nan_to_num(batch[0]).astype(floatX))
                self.yi.set_value(np.nan_to_num(batch[1]).astype(floatX))
                batch_cost += self.train_func()
            print epoch, batch_cost / n_batches

    def predict(self, data):
        self.xi.set_value(data)
        return self.predict_func()
