import os
import numpy as np
import tensorflow as tf
import random
from os import path, makedirs
from .Util import Dataset, eva_R
import json

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('name [MLP]')
    print('n_inputs [1]')
    print('n_steps {x:5}')
    print('input_features [x]')
    print('n_preds [1]')
    print('n_outputs [1]')
    print('output_feature [x]')
    print('n_hiddens [100, 300]')
    print('ckpt_path [../data/MLP/ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)
    print('init_lr [1e-3]')
    print('n_epochs [100]')
    print('batch_size [32]')
    print('ckpt [False]')
    print('L2_weight [0.1]')
    print('decay_power [1.75]')
    print('training_sample_num [1000]')
    print('device [/gpu:0]')

class Model:
    def __init__(self):
        self.name = 'MLP'
        self.n_inputs = 1
        self.n_steps = {'x':5}
        self.input_features = ['x']
        self.n_preds = 1
        self.n_outputs = 1
        self.output_feature = 'x'
        self.n_hiddens = [100, 300]
        self.ckpt_path = '../data/MLP/ckpt'
        
        self.init_lr = 1e-3
        self.n_epochs = 100
        self.batch_size = 32
        self.ckpt = False
        self.L2_weight = 0.1
        self.decay_power = 1.75
        self.training_sample_num = 1000
        self.device = '/gpu:0'
        self.graph = None
        
    def set_model_params(self, params):
        if 'name' in params:
            self.name = params['name']
        if 'n_inputs' in params:
            self.n_inputs = params['n_inputs']
        if 'n_steps' in params:
            self.n_steps = params['n_steps']
        if 'input_features' in params:
            self.input_features = params['input_features']
        if 'n_preds' in params:
            self.n_preds = params['n_preds']
        if 'n_outputs' in params:
            self.n_outputs = params['n_outputs']
        if 'output_feature' in params:
            self.output_feature = params['output_feature']
        if 'n_hiddens' in params:
            self.n_hiddens = params['n_hiddens']
        if 'ckpt_path' in params:
            self.ckpt_path = params['ckpt_path']
        self.__build_MLP()
    
    def save_model_params(self, save_path):
        model_params = {}
        model_params['name'] = self.name
        model_params['n_inputs'] = self.n_inputs
        model_params['n_steps'] = self.n_steps
        model_params['input_features'] = self.input_features
        model_params['n_preds'] = self.n_preds
        model_params['n_outputs'] = self.n_outputs
        model_params['output_feature'] = self.output_feature
        model_params['n_hiddens'] = self.n_hiddens
        model_params['ckpt_path'] = self.ckpt_path
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
        print('n_inputs:' + str(self.n_inputs))
        print('n_steps:' + str(self.n_steps))
        print('input_features:' + str(self.input_features))
        print('n_preds:' + str(self.n_preds))
        print('n_outputs:' + str(self.n_outputs))
        print('output_feature:' + str(self.output_feature))
        print('n_hiddens:' + str(self.n_hiddens))
        print('ckpt_path:' + str(self.ckpt_path))
        
    def set_training_params(self, params):
        if 'L2_weight' in params:
            self.lr_decay = params['L2_weight']
        if 'init_lr' in params:
            self.init_lr = params['init_lr']
        if 'n_epochs' in params:
            self.n_epochs = params['n_epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'ckpt' in params:
            self.ckpt = params['ckpt']
        if 'decay_power' in params:
            self.decay_power = params['decay_power']
        if 'training_sample_num' in params:
            self.training_sample_num = params['training_sample_num']
        if 'device' in params:
            self.device = params['device']
    def get_val_data(self, train_data, test_data):
        n_steps = max(self.n_steps.values())
        return [\
                train_item[-n_steps:]\
                + test_item\
                for train_item, test_item\
                in zip(train_data, test_data)
               ]
    def create_set(self, datalist):
        data_mats = []
        outputs = {}
        n_steps = max(self.n_steps.values())
        for feature in self.input_features:
            data_mat = generate_data_matrix(
                datalist[feature],
                n_steps,
                self.n_preds)
            data_mats.append(
                data_mat[:, -self.n_steps[feature] - 1:-1]
            )
            outputs[feature] = data_mat[:, -1]
        X = np.concatenate(data_mats, axis = 1)
        y = outputs[self.output_feature]
        return Dataset(X, y)
    
    def train(
        self,
        train_set,
        val_set = None):
        lr = self.init_lr
        train_X = train_set.X
        train_y = np.reshape(train_set.y, (-1, 1))
        if not val_set is None:
            val_X = val_set.X
            val_y = np.reshape(val_set.y, (-1, 1))
        if self.graph is None:
            self.__build_MLP()
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
            num_iter = self.n_epochs * (train_X.shape[0] // self.batch_size)
            for epoch in range(self.n_epochs):
                for i in range(0, train_X.shape[0], self.batch_size):
                    X_batch = train_X[i:i + self.batch_size, :]
                    y_batch = train_y[i:i + self.batch_size]
                    sess.run(
                        self.training_ops,
                        feed_dict = {
                            self.X:X_batch,
                            self.y:y_batch,
                            self.max_iters:num_iter,
                            self.training:True
                        }
                    )
  
                sample_num = self.training_sample_num
                if not val_set is None:
                    sample_num = val_X.shape[0]
                sampled_idx = random.sample(
                    range(train_X.shape[0]),
                    min(sample_num, train_X.shape[0])
                )
                sampled_X = train_X[sampled_idx, :]
                sampled_y = train_y[sampled_idx]
                train_R = sess.run(
                    self.Rsquare,
                    feed_dict = {
                        self.X:sampled_X,
                        self.y:sampled_y,
                        self.training:False
                    }
                )
                if not val_set is None:
                    val_R = sess.run(
                        self.Rsquare,
                        feed_dict = {
                            self.X:val_X,
                            self.y:val_y,
                            self.training:False
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
            if not path.exists(self.ckpt_path):
                makedirs(self.ckpt_path)
            save_path = self.saver.save(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
            print("Model saved in path: %s" % save_path)
            print('-' * 56)
    
    def get_preds(self, X):
        if self.graph is None:
            self.__build_MLP()
        with tf.Session(graph = self.graph) as sess:
            self.saver.restore(sess, path.join(self.ckpt_path, self.name + '.ckpt'))
            results = sess.run(
                self.output,
                feed_dict = {
                    self.X:X,
                    self.training:False
                }
            )
        return np.reshape(results, (-1, ))
    def __build_MLP(self):
        """---------------------------Build Phase---------------------------"""
        self.n_inputs = 0
        for feature in self.n_steps:
            self.n_inputs += self.n_steps[feature]
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(
                tf.float32,
                shape = (None, self.n_inputs),
                name = 'X')
            self.y = tf.placeholder(
                tf.float32,
                shape = (None, self.n_outputs),
                name = 'y')
            self.training = tf.placeholder_with_default(
                False,
                shape = (),
                name = 'training')

            last_output = self.X
            self.weights = []
            #create hidden layers with specified number of hidden neurons
            for i, n_hidden in enumerate(self.n_hiddens):
                if n_hidden == 0:
                    continue
                last_output = tf.layers.dense(
                    last_output,
                    n_hidden,
                    name = 'hidden_' + str(i),
                    activation = tf.nn.relu)
                self.weights.append(
                    self.graph.get_tensor_by_name(
                        'hidden_' + str(i) + '/kernel:0'
                    )
                )
           
            self.output = tf.layers.dense(
                last_output,
                self.n_outputs,
                name = 'output',
                activation = None)
            self.weights.append(
                self.graph.get_tensor_by_name('output/kernel:0')
            )

            #training
            self.loss = tf.multiply(
                tf.reduce_mean(
                    tf.square(
                        tf.subtract(self.y, self.output)
                    )
                ),
                0.5
            )
            
            #regularizer loss
            self.reg_loss = tf.constant(0.0)
            for w in self.weights:
                self.reg_loss += tf.reduce_sum(tf.square(w))
            #count the number of executed iterations
            self.global_steps = tf.Variable(
                0,
                trainable=False,
                dtype=tf.float32,
                name = 'global_step'
            )
            #maximum number of iterations
            self.max_iters = tf.placeholder(
                tf.float32,
                name = 'max_iter'
            )
            #polynomial decay learning rate
            learning_rate = self.__poly_decay(
                self.init_lr,
                self.global_steps,
                self.max_iters,
                self.decay_power)
            #learning_rate=init_learning_rate
            #momentum optimizer
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=0.9,
                beta2=0.999
            )
            self.total_loss = tf.add(
                self.loss,
                self.L2_weight * self.reg_loss
            )
            
            self.training_ops = optimizer.minimize(
                self.total_loss,
                global_step = self.global_steps,
                name='training_op'
            )
            
            #evaluate the MSE and R squared value for whole predictions
            residual_loss = tf.reduce_sum(
                tf.square(tf.subtract(self.y, self.output))
            )
            total_loss = tf.reduce_sum(
                tf.square(
                    tf.subtract(self.y, tf.reduce_mean(self.y))
                )
            )
            self.Rsquare = tf.subtract(
                1.0,
                tf.div(residual_loss, total_loss)
            )
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    def __poly_decay(self, init_lr, global_iter, max_iter, power):
        decay_rate = tf.subtract(
            1.0,
            tf.clip_by_value(
                tf.div(global_iter, max_iter), 0, 1
            )
        )
        return tf.multiply(init_lr, tf.pow(decay_rate, power))
    
def generate_data_matrix(data, n_steps, n_preds):
    m = len(data)
    data_mat = []
    for i in range(m):
        n = len(data[i])
        n_slides = n - n_steps - n_preds + 1
        mat = np.zeros((n_slides, n_steps + 1))
        for j in range(n_slides):
            mat[j, :-1] = data[i][j:j + n_steps]
            mat[j, -1] = data[i][j + n_steps + n_preds - 1]
        data_mat.append(mat)
    return np.concatenate(data_mat, axis = 0) 