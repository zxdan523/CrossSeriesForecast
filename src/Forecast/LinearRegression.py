import os
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import random
from os import path, makedirs
from .Util import Dataset, eva_R
import json

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('name [LR]')
    print('n_inputs [1]')
    print('n_steps {x:5}')
    print('input_features [x]')
    print('n_preds [1]')
    print('n_outputs [1]')
    print('output_feature [x]')
    print('intercept [False]')
    print('normalize [False]')
    print('ckpt_path: [../data/LR/ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)
    print('n_jobs [None]')

class Model:
    def __init__(self):
        self.name = 'LR'
        self.n_inputs = 1
        self.n_steps = {'x':5}
        self.input_features = ['x']
        self.n_preds = 1
        self.n_outputs = 1
        self.output_feature = 'x'
        self.intercept = False
        self.normalize = False
        self.n_jobs = None
        self.model = None
        self.ckpt_path = '../data/LR/ckpt'
        
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
        if 'intercept' in params:
            self.intercept = params['intercept']
        if 'normalize' in params:
            self.normalize = params['normalize']
        if 'ckpt_path' in params:
            self.ckpt_path = params['ckpt_path']
        self.model = None
    
    def save_model_params(self, save_path):
        model_params = {}
        model_params['name'] = self.name
        model_params['n_inputs'] = self.n_inputs
        model_params['n_steps'] = self.n_steps
        model_params['input_features'] = self.input_features
        model_params['n_preds'] = self.n_preds
        model_params['n_outputs'] = self.n_outputs
        model_params['output_feature'] = self.output_feature
        model_params['intercept'] = self.intercept
        model_params['normalize'] = self.normalize
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
        print('intercept:' + str(self.intercept))
        print('normalize:' + str(self.normalize))
        print('ckpt_path:' + str(self.ckpt_path))
        
    def set_training_params(self, params):
        if 'n_jobs' in params:
            self.n_jobs = params['n_jobs']
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
        train_X = train_set.X
        train_y = train_set.y
        self.model = LinearRegression(
            fit_intercept = self.intercept,
            normalize = self.normalize
        ).fit(
            train_X,
            train_y
        )
        train_pred = self.model.predict(train_X)
        train_R = eva_R(train_pred, train_y)
        if not val_set is None:
            val_X = val_set.X
            val_y = val_set.y
            val_pred = self.model.predict(val_X)
            val_R = eva_R(val_pred, val_y)
        print('Train R^2: {0} and Test R^2: {1}'.format(train_R, val_R))
        if not path.exists(self.ckpt_path):
            makedirs(self.ckpt_path)
            
        dump(self.model, path.join(self.ckpt_path, self.name + '.joblib'))
        print('Save Model to ' + path.join(self.ckpt_path, self.name + '.joblib'))
            
    def get_preds_with_horizon(self, x, horizon):
        pass
    
    def get_preds(self, X):
        if self.model is None:
            self.model = load(path.join(self.ckpt_path, self.name + '.joblib'))
        results = self.model.predict(X)
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