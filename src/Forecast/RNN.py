import os
import numpy as np
import tensorflow as tf
import random
from os import path, makedirs
from .Util import Dataset

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('n_neurons [100]')
    print('n_inputs [1]')
    print('n_outputs [1]')
    print('ckpt_path [../data/RNN_ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)
    print('init_lr [1e-3]')
    print('n_epochs [100]')
    print('batch_size [32]')
    print('lr_decay [0.1]')
    print('lr_decay_steps [2]')
    print('lr_schedule [learning rate schedule function]')
    print('ckpt [False]')
    print('training_sample_num [1000]')

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
    def __init__(self, name, n_steps, n_preds, params):
        self.name = name
        self.n_steps = n_steps
        self.n_preds = n_preds
        self.n_inputs = 1
        if 'n_input' in params:
            self.n_inputs = params['n_inputs']
        self.n_neurons = 100
        if 'n_neurons' in params:
            self.n_neurons = params['n_neurons']
        self.ckpt_path = '../data/RNN_ckpt'
        if 'ckpt_path' in params:
            self.ckpt_path = params['ckpt_path']
        self.n_outputs = 1
        if 'n_outputs' in params:
            self.n_outputs = params['n_outputs']
        self.lr_decay = 0.1
        self.lr_decay_steps = 2
        self.init_lr = 1e-3
        self.n_epochs = 100
        self.batch_size = 32
        self.lr_schedule = self.__lr_schedule
        self.ckpt = False
        self.training_sample_num = 1000
        
    def set_training_params(self, params):
        if 'lr_decay' in params:
            self.lr_decay = params['lr_decay']
        if 'lr_decay_steps' in params:
            self.lr_decay_steps = params['lr_decay_steps']
        if 'init_lr' in params:
            self.init_lr = params['init_lr']
        if 'n_epochs' in params:
            self.n_epochs = params['n_epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'lr_schedule' in params:
            self.lr_schedule = params['lr_schedule']
        if 'ckpt' in params:
            self.ckpt = True
        if 'training_sample_num' in params:
            self.training_sample_num = params['training_sample_num']
       
    def create_set(self, main_data, datalist = []):
        datalist = [main_data] + datalist
        data_mats = []
        for data in datalist:
            data_mat = generate_data_matrix(data, self.n_steps, self.n_preds)
            data_mats.append(data_mat)
        data_mat = np.concatenate(data_mats, axis = 2)
        X = data_mat[:, :-self.n_preds,:]
        y = data_mat[:, self.n_preds:,:]
        return Dataset(X, y)
    
    def train(
        self,
        train_set,
        val_set = None):
        lr = self.init_lr
        train_X = train_set.X
        train_y = train_set.y
        if not val_set is None:
            val_X = val_set.X
            val_y = val_set.y
        self.n_inputs = train_X.shape[2]
        self.__build_RNN()
        with tf.Session(graph = self.graph) as sess:
            if not self.ckpt:
                init = tf.global_variables_initializer()
                init.run()
            else:
                self.saver.restore(
                    sess,
                    path.join(self.ckpt_path, self.name + '.ckpt')
                )
            print('Training Model: ' + self.name)
            print('-' * 56)
            last_training_error = None
            for epoch in range(self.n_epochs):
                lr = self.lr_schedule(lr, self.n_epochs, epoch)
                for i in range(0, train_X.shape[0], self.batch_size):
                    X_batch = train_X[i:i + self.batch_size, :, :]
                    y_batch = train_y[i:i + self.batch_size, :, :]
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
                
                sample_num = self.training_sample_num
                if not val_set is None:
                    sample_num = val_X.shape[0]
                sampled_idx = random.sample(
                    range(train_X.shape[0]),
                    min(sample_num, train_X.shape[0])
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
                if not val_set is None:
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
                else:
                    print('{0}\t|train R^2:{1:.5f}'.format(epoch, train_R))
                if lr < 1e-5 and np.abs(train_R - last_training_error) < 1e-6:
                    break
                last_training_error = lr
            if not path.exists(self.ckpt_path):
                makedirs(self.ckpt_path)
            save_path = self.saver.save(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
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
        
    def __lr_schedule(self, lr, n_epochs, epoch):
        if epoch > 0 and epoch % (n_epochs // self.lr_decay_steps) == 0:
            lr *= self.lr_decay
            print('learning rate decrease to', lr)
        return lr
    
    def __build_RNN(self):
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
    
def generate_data_matrix(data, n_steps, n_preds):
    m = len(data)
    data_mat = []
    for i in range(m):
        n = len(data[i])
        n_slides = n - n_steps - n_preds + 1
        mat = np.zeros((n_slides, n_steps + n_preds, 1))
        for j in range(n_slides):
            mat[j, :, 0] = data[i][j:j + n_steps + n_preds]
        data_mat.append(mat)
    return np.concatenate(data_mat, axis = 0) 