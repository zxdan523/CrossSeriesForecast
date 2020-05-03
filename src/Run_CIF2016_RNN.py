from Forecast.Util import *
from Forecast import RNN
from Forecast import CIF2016
import numpy as np
from os import path, makedirs

print('Loading Data ...')
train_data = CIF2016.import_data('../data/cif2016.txt', 3)
test_data = CIF2016.import_data('../data/cif2016-test.txt', 1)
name = 'cif2016_RNN'

print('Setting Parameters ...')
groups = [
    [i for i in range(len(train_data)//2)],
    [i for i in range(len(train_data)//2, len(train_data))]
]

init_model_params = {
    'input_features': ['sale'],
    'output_features': ['sale']
}

train_params = {
    'n_epochs': 30
}

model_params_list = [
    ('n_steps', list(range(5, 6))),
    ('n_neurons', [100, 300])
]

perform_path = '../data/CIF2016/RNN/performances'

print('Runing Grid Search ...')

for idx, group in enumerate(groups):
    rnn = RNN.Model()
    rnn.set_model_params(init_model_params)
    rnn.set_training_params(train_params)
    group_train_data = [train_data[idx] for idx in group]
    group_test_data = [train_data[idx] for idx in group]
    train_datalist = {'sale': group_train_data}
    test_datalist = {'sale': group_test_data}
    if not path.exists(perform_path):
        makedirs(perform_path)
    grid_search_model_params(
        rnn,
        name,
        {},
        model_params_list,
        train_datalist,
        test_datalist,
        '../data/CIF2016/RNN/performances/results_' + str(idx) + '.csv',
        '../data/CIF2016/RNN/model_params'
    )
    print('Finished Group ' + str(idx))