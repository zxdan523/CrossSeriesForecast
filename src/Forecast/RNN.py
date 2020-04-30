import os
import numpy as np
import tensorflow as tf
import random
from os import path, makedirs
from .Util import generate_data_matrix

class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device
    @property
    def state_size(self):
        return self._cell.state_size
    @property
    def output_size(self):
        return self._cell.output_size
    def __call__(self,inputs,state,scope = None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)
        
class RNN:
    def __init__(self, name, n_steps, n_preds, n_inputs, n_neurons, n_outputs):
        self.name = name
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_preds = n_preds
        self.lr_decay = 0.1
        self.lr_decay_steps = 2
        self.build_RNN()
    def set_decay_params(self, lr_decay, lr_decay_steps):
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
    def lr_schedule(self, lr, n_epochs, epoch):
        if epoch > 0 and epoch % (n_epochs // self.lr_decay_steps) == 0:
            lr *= self.lr_decay
            print('learning rate decrease to', lr)
        return lr
    def build_RNN(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            # --------------- Building Phase ---------------
            tf.variable_scope(
                "rnn",
                initializer = tf.variance_scaling_initializer()
            )
            self.X = tf.placeholder(
                tf.float32,
                [None, self.n_steps, self.n_inputs],
                name = "X"
            )
            self.y = tf.placeholder(
                tf.float32, 
                [None, self.n_steps, self.n_inputs],
                name="y"
            )
            self.cells = tf.contrib.rnn.BasicRNNCell(
                num_units = self.n_neurons,
                activation = tf.nn.relu,
                reuse = tf.AUTO_REUSE
            )
            self.gpu_cells = DeviceCellWrapper(
                '/gpu:0',
                self.cells
            )
            self.rnn_outputs, self.states = tf.nn.dynamic_rnn(
                self.gpu_cells,
                self.X,
                dtype=tf.float32
            )
            self.stacked_rnn_outputs = tf.reshape(
                self.rnn_outputs,
                [-1, self.n_neurons]
            )
            self.stacked_outputs = tf.layers.dense(
                self.stacked_rnn_outputs,
                self.n_outputs
            )
            self.outputs = tf.reshape(
                self.stacked_outputs,
                [-1, self.n_steps, self.n_outputs],
                name = "outputs"
            )
            
            self.predict = self.outputs[:,-1,0]
            
            self.ys = self.y[:,-1,0]
            
            # --------------- Training Phase ---------------
            self.init_lr = tf.placeholder(
                tf.float32,
                shape=[],
                name="init_lr"
            )
            
            self.loss = tf.reduce_mean(
                tf.square(
                    self.predict - self.ys
                )
            )
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate = self.init_lr,
                beta1 = 0.9,
                beta2 = 0.999
            )
            self.training_ops = self.optimizer.minimize(
                self.loss,
                name = "training_ops"
            )
            
            # --------------- Test Phase ---------------
            with tf.name_scope('eval'):
                self.total_mse = tf.multiply(
                    tf.reduce_mean(
                        tf.square(
                            tf.subtract(self.ys, self.predict)
                        )
                    ),
                    0.5,
                    name = 'total_mse'
                )
                residual_loss = tf.reduce_sum(
                    tf.square(
                        tf.subtract(self.ys, self.predict)
                    )
                )
                total_loss = tf.reduce_sum(
                    tf.square(
                        tf.subtract(self.ys, tf.reduce_mean(self.ys))
                    )
                )
                self.Rsquare = tf.subtract(
                    1.0,
                    tf.div(residual_loss, total_loss),
                    name = "Rsquare"
                )
            self.saver = tf.train.Saver()

    def train(
        self,
        train_set,
        val_set,
        n_epochs,
        batch_size,
        init_lr,
        ckpt = False):
        
        lr = init_lr
        train_X = train_set.X
        train_y = train_set.y
        val_X = val_set.X
        val_y = val_set.y
        with tf.Session(graph = self.graph) as sess:
            if not ckpt:
                init = tf.global_variables_initializer()
                init.run()
            else:
                self.saver.restore(
                    sess,
                    "../data/RNN_ckpt/" + self.name + '.ckpt'
                )
            print('Training Model: ' + self.name)
            print('-' * 56)
            train_Rs = []
            for epoch in range(n_epochs):
                lr = self.lr_schedule(lr, n_epochs, epoch)
                for i in range(0, train_X.shape[0], batch_size):
                    X_batch = train_X[i:i+batch_size, :, :]
                    y_batch = train_y[i:i+batch_size, :, :]
                    sess.run(
                        self.training_ops,
                        feed_dict = {
                            self.X:X_batch,
                            self.y:y_batch,
                            self.init_lr:lr
                        }
                    )
                    train_R = sess.run(
                        self.Rsquare,
                        feed_dict = {
                            self.X:X_batch,
                            self.y:y_batch,
                        }
                    )
                
                sampled_idx = random.sample(
                    range(train_X.shape[0]),
                    val_X.shape[0]
                )
                sampled_X = train_X[sampled_idx, :, :]
                sampled_y = train_y[sampled_idx, :, :]
                train_R = sess.run(
                    self.Rsquare,
                    feed_dict = {
                        self.X:sampled_X,
                        self.y:sampled_y,
                    }
                )
                val_R = sess.run(
                    self.Rsquare,
                    feed_dict = {
                        self.X:val_X,
                        self.y:val_y,
                    }
                )

                print(
                    '{0}\t|train R^2:{1:.5f}\t|validate R^2:{2:.5f}'.format(
                    epoch,
                    train_R,
                    val_R
                    )
                )
            if not path.exists('../data/RNN_ckpt'):
                makedirs('../data/RNN_ckpt')
            save_path = self.saver.save(sess, "../data/RNN_ckpt/" + self.name + '.ckpt')
            print("Model saved in path: %s" % save_path)
            print('-' * 56)
    def get_preds_with_horizon(self, x, horizon):
        tf.logging.set_verbosity(tf.logging.ERROR)
        results = []
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, "../data/RNN_ckpt/" + self.name + '.ckpt')
            for h in range(horizon):
                hist = np.zeros((1, self.n_steps, self.n_inputs))
                if h < self.n_steps:
                    hist[0, :, 0] = x[-self.n_steps + h:] + results[:h]
                else:
                    hist[0, :, 0] = results[-self.n_steps:]
                result = sess.run(
                    self.predict,
                    feed_dict = {
                        self.X:hist
                    }
                )
                results.append(result[0])
        tf.logging.set_verbosity(tf.logging.DEBUG)
        return results
    def get_preds(self, X):
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, "../data/RNN_ckpt/" + self.name + '.ckpt')
            results = sess.run(
                self.predict,
                feed_dict = {
                    self.X:X
                }
            )
        return results
    def get_eval(self, var, feed_dict):
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, "../data/RNN_ckpt/" + self.name + '.ckpt')
            results = sess.run(
                var,
                feed_dict = feed_dict
            )
        return results