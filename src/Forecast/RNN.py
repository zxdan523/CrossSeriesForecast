import os
import numpy as np
import tensorflow as tf
import random
from os import path, makedirs
from .Util import Dataset, eva_R
import json

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('name [RNN]')
    print('n_steps [5]')
    print('n_preds [1]')
    print('n_neurons [100]')
    print('cell_type [basic]')
    print('n_layers [1]')
    print('n_gpus [0]')
    print('n_inputs [1]')
    print('input_features [x]')
    print('n_outputs [1]')
    print('output_feature [x]')
    print('ckpt_path [../data/RNN_ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)
    print('init_lr [1e-3]')
    print('n_epochs [100]')
    print('batch_size [32]')
    print('lr_decay [0.1]')
    print('lr_decay_steps [2]')
    print('lr_schedule [learning rate schedule function]')
    print('ckpt [False]')
    print('drop_rate [1.0]')

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

class Model:
    def __init__(self):
        self.name = 'RNN'
        self.n_steps = 5
        self.n_preds = 1
        self.n_inputs = 1
        self.input_features = ['x']
        self.n_neurons = 100
        self.cell_type = 'basic'
        self.n_layers = 1
        self.n_gpus = 0
        self.ckpt_path = '../data/RNN_ckpt'
        self.n_outputs = 1
        self.output_feature = 'x'
        self.lr_decay = 0.1
        self.lr_decay_steps = 2
        self.init_lr = 1e-3
        self.n_epochs = 100
        self.batch_size = 32
        self.lr_schedule = self.__lr_schedule
        self.ckpt = False
        self.drop_rate = 1.0
        self.graph = None

    def set_model_params(self, params):
        if 'name' in params:
            self.name = params['name']
        if 'n_steps' in params:
            self.n_steps = params['n_steps']
        if 'n_preds' in params:
            self.n_preds = params['n_preds']
        if 'n_inputs' in params:
            self.n_inputs = params['n_inputs']
        if 'input_features' in params:
            self.input_features = params['input_features']
            self.n_inputs = len(self.input_features)
        if 'n_neurons' in params:
            self.n_neurons = params['n_neurons']
        if 'cell_type' in params:
            self.cell_type = params['cell_type']
        if 'n_layers' in params:
            self.n_layers = params['n_layers']
        if 'n_gpus' in params:
            self.n_gpus = params['n_gpus']
        if 'ckpt_path' in params:
            self.ckpt_path = params['ckpt_path']
        if 'n_outputs' in params:
            self.n_outputs = params['n_outputs']
        if 'output_feature' in params:
            self.output_feature = params['output_feature']
        self.__build_RNN()

    def save_model_params(self, save_path):
        model_params = {}
        model_params['name'] = self.name
        model_params['n_steps'] = self.n_steps
        model_params['n_preds'] = self.n_preds
        model_params['n_inputs'] = self.n_inputs
        model_params['input_features'] = self.input_features
        model_params['n_neurons'] = self.n_neurons
        model_params['cell_type'] = self.cell_type
        model_params['n_layers'] = self.n_layers
        model_params['n_gpus'] = self.n_gpus
        model_params['ckpt_path'] = self.ckpt_path
        model_params['n_outputs'] = self.n_outputs
        model_params['output_feature'] = self.output_feature
        if not path.exists(save_path):
            makedirs(save_path)
        with open(path.join(save_path, self.name + '.json'), 'w') as json_file:
            json.dump(model_params, json_file)
        print('The model is saved to ' + path.join(save_path, self.name + '.json'))

    def load_model_params(self, save_file):
        print('Load model from file ' + save_file)
        with open(save_file) as json_file:
            model_params = json.load(json_file)
        self.set_model_params(model_params)

    def show_model_params(self):
        print('-' * 20 + ' Model Params ' + '-' * 20)
        print('name:' + str(self.name))
        print('n_steps:' + str(self.n_steps))
        print('n_preds:' + str(self.n_preds))
        print('n_neurons:' + str(self.n_neurons))
        print('cell_type:' + str(self.cell_type))
        print('n_gpus:' + str(self.n_gpus))
        print('n_layers:' + str(self.n_layers))
        print('n_inputs:' + str(self.n_inputs))
        print('input_features:' + str(self.input_features))
        print('n_outputs:' + str(self.n_outputs))
        print('ckpt_path:' + str(self.ckpt_path))
        print('output_feature:' + str(self.output_feature))

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
            self.ckpt = params['ckpt']
        if 'drop_rate' in params:
            self.drop_rate = params['drop_rate']
    def get_val_data(self, train_data, test_data):
        return [\
                train_item[-self.n_steps:]\
                + test_item\
                for train_item, test_item\
                in zip(train_data, test_data)
               ]
    def create_set(self, datalist):
        data_mats = []
        for feature in self.input_features:
            data_mat = generate_data_matrix(
                datalist[feature],
                self.n_steps,
                self.n_preds)
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
        if self.graph is None:
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
                            self.lr:lr,
                            self.keep_prob:self.drop_rate,
                        }
                    )

                train_preds = []
                for i in range(0, train_X.shape[0], self.batch_size):
                    X_batch = train_X[i:i + self.batch_size, :, :]
                    pred = sess.run(
                        self.predict,
                        feed_dict = {
                            self.X:X_batch,
                            self.keep_prob:1.0,
                        }
                    )
                    train_preds.append(pred)
                train_R = eva_R(np.concatenate(train_preds), train_y[:, -1, 0])
                if not val_set is None:
                    val_preds = []
                    for i in range(0, val_X.shape[0], self.batch_size):
                        X_batch = val_X[i:i + self.batch_size, :, :]
                        pred = sess.run(
                            self.predict,
                            feed_dict = {
                                self.X:X_batch,
                                self.keep_prob:1.0,
                            }
                        )
                        val_preds.append(pred)
                    val_R = eva_R(np.concatenate(val_preds), val_y[:, -1, 0])

                    print(
                        '{0}\t|train R^2:{1:.5f}\t|validate R^2:{2:.5f}'.format(
                        epoch,
                        train_R,
                        val_R
                        )
                    )
                else:
                    print('{0}\t|train R^2:{1:.5f}'.format(epoch, train_R))
            if not path.exists(self.ckpt_path):
                makedirs(self.ckpt_path)
            save_path = self.saver.save(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
            print("Model saved in path: %s" % save_path)
            print('-' * 56)

    def get_preds_with_horizon(self, x, horizon):
        tf.logging.set_verbosity(tf.logging.ERROR)
        results = []
        if self.graph is None:
            self.__build_RNN()
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
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
        if self.graph is None:
            self.__build_RNN()
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
            preds = []
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size, :, :]
                pred = sess.run(
                    self.predict,
                    feed_dict = {
                        self.X:X_batch,
                        self.keep_prob:1.0,
                    }
                )
                preds.append(pred)
        return np.concatenate(preds)

    def get_eval(self, var, feed_dict):
        if self.graph is None:
            self.__build_RNN()
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
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

    def __build_cells(self):
        self.keep_prob = tf.placeholder_with_default(1.0, shape = ())
        if self.cell_type == 'basic':
            self.RNN_factory = tf.contrib.rnn.BasicRNNCell
        elif self.cell_type == 'lstm':
            self.RNN_factory = tf.contrib.rnn.LSTMCell
        if self.n_gpus == 0:
            cells = [
                self.RNN_factory(
                    num_units = self.n_neurons,
                    activation = tf.nn.relu
                ) for _ in range(self.n_layers)\
            ]
        elif self.n_gpus >= self.n_layers:
            cells = [
                DeviceCellWrapper(
                    '/gpu:' + str(i),
                    self.RNN_factory(
                        num_units = self.n_neurons,
                        activation = tf.nn.relu
                    )
                ) for i in range(self.n_layers)\
            ]
        else:
            cells = []
            gpu_no = 0
            num_layer_per_gpu = self.n_layers // self.n_gpus
            for layer in range(self.n_layers):
                if (layer + 1) % num_layer_per_gpu == 0:
                    if gpu_no < self.n_gpus - 1:
                        gpu_no += 1
                cells.append(
                    DeviceCellWrapper(
                        '/gpu:' + str(gpu_no),
                        self.RNN_factory(
                            num_units = self.n_neurons,
                            activation = tf.nn.relu
                        )
                    )
                )
                
        cell_drops = [
            tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob = self.keep_prob
            ) for cell in cells\
        ]
        return cell_drops
                    
    def __build_RNN(self):
        self.graph = tf.Graph()
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

            self.cells = self.__build_cells()
            self.multi_layer_cells = tf.contrib.rnn.MultiRNNCell(self.cells)
            self.rnn_outputs, self.states = tf.nn.dynamic_rnn(
                self.multi_layer_cells,
                self.X,
                swap_memory = True,
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
            self.lr = tf.placeholder(
                tf.float32,
                shape=[],
                name="lr"
            )

            self.loss = tf.reduce_mean(
                tf.square(
                    self.predict - self.ys
                )
            )
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate = self.lr,
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
