import os
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.iolib.smpickle import load_pickle
import random
from os import path, makedirs
from .Util import Dataset, eva_R
import json

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('name [MA]')
    print('n_inputs [1]')
    print('n_steps {x:5}')
    print('input_features [x]')
    print('n_preds [1]')
    print('n_outputs [1]')
    print('output_feature [x]')
    print('smoothing [2]')
    print('ckpt_path: [../data/MA/ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)
    print('skip_training [False]')

class Model:
    def __init__(self):
        self.name = 'MA'
        self.n_inputs = 1
        self.n_steps = 1
        self.input_features = ['x']
        self.n_preds = 1
        self.n_outputs = 1
        self.output_feature = 'x'
        self.smoothing = 2
        self.ckpt_path = '../data/MA/ckpt'
        self.skip_training = False
        
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
        if 'smoothing' in params:
            self.trend = params['smoothing']
        if 'ckpt_path' in params:
            self.ckpt_path = params['ckpt_path']
        if 'skip_training' in params:
            self.skip_training = params['skip_training']
    
    def save_model_params(self, save_path):
        model_params = {}
        model_params['name'] = self.name
        model_params['n_inputs'] = self.n_inputs
        model_params['n_steps'] = self.n_steps
        model_params['input_features'] = self.input_features
        model_params['n_preds'] = self.n_preds
        model_params['n_outputs'] = self.n_outputs
        model_params['output_feature'] = self.output_feature
        model_params['smoothing'] = self.smoothing
        model_params['ckpt_path'] = self.ckpt_path
        model_params['skip_training'] = self.skip_training
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
        print('smoothing:' + str(self.smoothing))
        print('ckpt_path:' + str(self.ckpt_path))
        print('skip_training' + str(self.skip_training))
        
    def set_training_params(self, params):
        if 'skip_training' in params:
            self.skip_training = params['skip_training'] 
    def get_val_data(self, train_data, test_data):
        n_steps = self.n_steps[self.output_feature]
        return [\
                train_item[-n_steps - self.n_preds + 1:]\
                + test_item\
                for train_item, test_item\
                in zip(train_data, test_data)
               ]
    def create_set(self, datalist):
        n_steps = self.n_steps[self.output_feature]
        data_mat = generate_data_matrix(
            datalist[self.output_feature],
            n_steps,
            self.n_preds)
        X = data_mat[:,:-1]
        y = data_mat[:, -1]
        return Dataset(X, y)
    
    def train(
        self,
        train_set,
        val_set = None):
        if self.skip_training:
            return
        train_preds = []
        for x in train_set.X:
            ema = 0
            for i, v in enumerate(x):
                k = self.smoothing / (1 + i + 1)
                ema = v * k + ema * (1 - k)
            N = len(x)
            for i in range(self.n_preds - 1):
                k = self.smoothing / (1 + i + N + 1)
                ema = ema * k + ema * (1 - k)
            train_preds.append(ema)
        train_R = eva_R(np.array(train_preds), train_set.y)    
        if not val_set is None:
            val_X = val_set.X
            val_y = val_set.y
            val_preds = []
            for x in val_X:
                ema = 0
                for i, v in enumerate(x):
                    k = self.smoothing / (1 + i + 1)
                    ema = v * k + ema * (1 - k)
                N = len(x)
                for i in range(self.n_preds - 1):
                    k = self.smoothing / (1 + i + N + 1)
                    ema = ema * k + ema * (1 - k)
                val_preds.append(ema)
            val_R = eva_R(np.array(val_preds), val_ys)
        print('Test R^2: {0}'.format(val_R))
            
    def get_preds_with_horizon(self, x, horizon):
        ema = 0
        for i, v in enumerate(x):
            k = self.smoothing / (1 + i + 1)
            ema = v * k + ema * (1 - k)
        N = len(x)
        for i in range(horizon - 1):
            k = self.smoothing / (1 + i + N + 1)
            ema = ema * k + ema * (1 - k)
        return ema
    
    def get_preds(self, X):
        results = []
        for x in X:
            ema = 0
            for i, v in enumerate(x):
                k = self.smoothing / (1 + i + 1)
                ema = v * k + ema * (1 - k)
            N = len(x)
            for i in range(self.n_preds - 1):
                k = self.smoothing / (1 + i + N + 1)
                ema = ema * k + ema * (1 - k)
            results.append(ema)
        return results
    
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