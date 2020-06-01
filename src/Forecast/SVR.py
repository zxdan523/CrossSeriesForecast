import os
import numpy as np
from sklearn.svm import SVR
from joblib import dump, load
import random
from os import path, makedirs
from .Util import Dataset, eva_R
import json

def show_param_list():
    print('-' * 20 + ' Model Params ' + '-' * 20)
    print('name [SVR]')
    print('n_inputs [1]')
    print('n_steps {x:5}')
    print('input_features [x]')
    print('n_preds [1]')
    print('n_outputs [1]')
    print('output_feature [x]')
    print('kernel [rbf]')
    print('degree [3]')
    print('gamma [scale]')
    print('tol [1e-3]')
    print('C [1.0]')
    print('epsilon [0.1]')
    print('shrinking [True]')
    print('max_iter [-1]')
    print('ckpt_path: [../data/SVR/ckpt]')
    print('-' * 20 + ' Train Params ' + '-' * 20)

class Model:
    def __init__(self):
        self.name = 'SVR'
        self.n_inputs = 1
        self.n_steps = {'x':5}
        self.input_features = ['x']
        self.n_preds = 1
        self.n_outputs = 1
        self.output_feature = 'x'
        self.kernel = 'rbf'
        self.degree = 3
        self.gamma = 'scale'
        self.tol = 1e-3
        self.C = 1.0
        self.epsilon = 0.1
        self.shrinking = True
        self.max_iter = -1
        self.ckpt_path = '../data/SVR/ckpt'
        self.model = None
        
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
        if 'kernel' in params:
            self.kernel = params['kernel']
        if 'degree' in params:
            self.degree = params['degree']
        if 'gamma' in params:
            self.gamma = params['gamma']
        if 'tol' in params:
            self.tol = params['tol']
        if 'C' in params:
            self.C = params['C']
        if 'epsilon' in params:
            self.epsilon = params['epsilon']
        if 'shrinking' in params:
            self.shrinking = params['shrinking']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
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
        model_params['kernel'] = self.kernel
        model_params['degree'] = self.degree
        model_params['gamma'] = self.gamma
        model_params['tol'] = self.tol
        model_params['C'] = self.C
        model_params['epsilon'] = self.epsilon
        model_params['shrinking'] = self.shrinking
        model_params['max_iter'] = self.max_iter
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
        print('kernel:' + str(self.kernel))
        print('degree:' + str(self.degree))
        print('gamma:' + str(self.gamma))
        print('tol:' + str(self.tol))
        print('C:' + str(self.C))
        print('epsilon:' + str(self.epsilon))
        print('shrinking:' + str(self.shrinking))
        print('max_iter:' + str(self.max_iter))
        print('ckpt_path:' + str(self.ckpt_path))
        
    def set_training_params(self, params):
        pass
    def get_val_data(self, train_data, test_data):
        n_steps = max(self.n_steps.values())
        return [\
                train_item[-n_steps - self.n_preds + 1:]\
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
        self.model = SVR(
            kernel = self.kernel,
            degree = self.degree,
            gamma = self.gamma,
            tol = self.tol,
            C = self.C,
            epsilon = self.epsilon,
            shrinking = self.shrinking,
            max_iter = self.max_iter
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