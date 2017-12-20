from __future__ import division, print_function
import copy
import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX


class SGD(object):
    """
    Stochastic gradient descent algorithm for training MLP instances
    Includes momentum, learning rate scheduling, early stopping.
    """

    def __init__(self, mlp, learning_rate=0.1,
                 momentum=0.5, batch_size=100, iterations=100,
                 patience=0, rate_decay_th=0.1):
        self.mlp = mlp
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.patience = patience
        self.do_validation = patience > 0
        self.rate_decay_th = rate_decay_th

        self.l_r = l_r = T.scalar('l_r', dtype=floatX)
        self.learning_rate_decay = 1

        self.training_error = []
        cost = mlp.cost / (2 * batch_size)
        self.gparams = [T.grad(cost, param) for param in mlp.params]
        self.param_updates = [
            theano.shared(value=np.zeros(
                param.get_value().shape,
                dtype=theano.config.floatX)
            )
            for param in mlp.params
        ]
        self.updates = []
        for param, gparam, uparam in zip(
                mlp.params,
                self.gparams,
                self.param_updates):

            self.updates.append(
                (param, param - self.l_r * uparam)
            )

            self.updates.append(
                (uparam, (momentum * uparam) + ((1. - momentum) * gparam))
            )
        n_bins = mlp.hidden_layers[0].W.shape.eval()[0]

        self.xi = theano.shared(
            name='xi',
            value=np.zeros((self.batch_size, n_bins), dtype=floatX),
            allow_downcast=True)

        self.yi = theano.shared(
            name='yi',
            value=np.zeros((self.batch_size, n_bins), dtype=floatX),
            allow_downcast=True)

        self.train_func = theano.function(
            inputs=[self.l_r],
            outputs=cost,
            updates=self.updates,
            allow_input_downcast=True,
            givens={mlp.input: self.xi, mlp.target: self.yi}
        )

        self.validation_func = theano.function(
            inputs=[],
            outputs=cost,
            allow_input_downcast=True,
            givens={mlp.input: self.xi, mlp.target: self.yi}
        )
        self.predict_func = theano.function(
            inputs=[],
            outputs=mlp.output,
            givens={mlp.input: self.xi},
            allow_input_downcast=True
        )

    def train(self, dataset):
        """
        Train the MLP passed to the constructor using a Dataset.
        """
        n_batches = dataset.num_batches(self.batch_size)
        patience = self.patience

        if self.do_validation:
            val_batches = int(n_batches / 5)  # 20% validation
            n_batches = n_batches - val_batches
            best_val_err = np.inf
        prev_err = np.inf

        for epoch in range(self.iterations):
            train_err = 0
            for batch in dataset.batcher(0, n_batches, self.batch_size):
                self.xi.set_value(np.nan_to_num(batch[0]).astype(floatX))
                self.yi.set_value(np.nan_to_num(batch[1]).astype(floatX))
                train_err += self.train_func(self.learning_rate)

            if self.do_validation:
                val_err = 0
                for batch in dataset.batcher(n_batches,
                                             n_batches + val_batches,
                                             self.batch_size):
                    self.xi.set_value(
                        np.nan_to_num(batch[0]).astype(floatX))
                    self.yi.set_value(
                        np.nan_to_num(batch[1]).astype(floatX))
                    val_err += self.validation_func()
                val_err = val_err / n_batches
                print("val_err", val_err, "best ", best_val_err)
                if val_err < best_val_err:
                    best_params = copy.deepcopy(self.mlp.params)
                    best_val_err = val_err
                    patience = self.patience
                else:
                    patience -= 1

            err = train_err / n_batches
            print(epoch, err)

            if self.do_validation and patience == 0:
                print("patience over, returning")
                self.mlp.params = best_params
                print("final validation error ", best_val_err)
                return
            print(err - prev_err)
            if prev_err - err < self.rate_decay_th:
                if self.learning_rate_decay == 1:
                    print("starting learning rate decay")
                self.learning_rate_decay = 0.9
            self.learning_rate = self.learning_rate * self.learning_rate_decay
            prev_err = err

    def predict(self, data):
        """
        Get mlp prediction from observation.
        """
        self.xi.set_value(data.astype(floatX))
        return self.predict_func()
