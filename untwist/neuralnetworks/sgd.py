import copy
import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

class SGD(object):

    def __init__(self, mlp, learning_rate = 0.1, 
        momentum = 0.5, batch_size = 100, iterations = 100,
        patience = 0, patience_increase = 0, improvement_threshold = 0):
        self.mlp = mlp
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        
        
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

        self.validation_func = theano.function(
            inputs = [],
            outputs = cost,            
            allow_input_downcast = True,
            givens = {mlp.input:self.xi, mlp.target:self.yi}
        )        
        self.predict_func = theano.function(
            inputs = [],
            outputs = mlp.output,
            givens = {mlp.input:self.xi},
            allow_input_downcast = True
        )

    def do_validation(self, iteration):        
        return self.patience > 0 and (iteration +1) % self.validation_frequency == 0

    def train(self, dataset):
        n_batches = dataset.X.shape[0] / self.batch_size            
        
        if self.patience > 0:            
            val_batches = int(n_batches / 10) # 10% validation            
            n_batches = n_batches - val_batches            
            self.validation_frequency = min(n_batches, self.patience/2)            
            best_val_loss = np.inf

        for epoch in range(self.iterations):
            batch_cost = 0
            for index in xrange(n_batches):
                iteration =  epoch  * n_batches + index
                if self.do_validation(iteration):
                    print "performing validation"
                    val_loss = 0
                    for val_index in range(n_batches, n_batches + val_batches):
                        val_batch = dataset.get_batch(val_index, self.batch_size)                               
                        self.xi.set_value(
                            np.nan_to_num(val_batch[0])
                        )
                        self.yi.set_value(
                            np.nan_to_num(val_batch[1]).astype(floatX)
                        )                                        
                        val_loss += self.validation_func()
                    print "val_loss", val_loss, "best ", best_val_loss  
                    if val_loss < best_val_loss:
                        if val_loss < best_val_loss * self.improvement_threshold:
                            self.patience = max(self.patience, iteration * self.patience_increase)
                            print "improved loss, patience", self.patience
                            best_params = copy.deepcopy(self.mlp.params) 
                            best_val_loss = val_loss
                batch = dataset.get_batch(index, self.batch_size)      
                self.xi.set_value(np.nan_to_num(batch[0]).astype(floatX))
                self.yi.set_value(np.nan_to_num(batch[1]).astype(floatX))

                batch_cost += self.train_func()
                if self.do_validation(iteration) and self.patience <= iteration:
                    self.mlp.params = best_params
                    return

            print epoch, batch_cost / n_batches

    def predict(self, data):
        self.xi.set_value(data)
        return self.predict_func()
