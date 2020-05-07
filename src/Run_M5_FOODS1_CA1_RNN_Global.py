from Forecast.Util import *
from Forecast import RNN
from Forecast import M5
import numpy as np
from os import path, makedirs

print('Loading Data ...')
sales, prices = M5.import_weekly_sales_and_prices('../data/m5-forecasting-accuracy')
dates = M5.import_weekly_dates('../data/m5-forecasting-accuracy')
name = 'M5_FOODS1_CA1'

dept_id = 'FOODS_1'
store_id = 'CA_1'
training_years = range(2012, 2015)
test_years = [2015]

base_path = '../data/M5/' + dept_id + '_' + store_id + '/RNN/global'
perform_path = path.join(base_path, 'performance')
model_path = path.join(base_path, 'models')
ckpt_path = path.join(base_path, 'ckpts')

category = {'dept_id': 'FOODS_1', 'store_id':'CA_1'}
train_dates = M5.select_dates(dates, range(2012, 2015), range(1, 13))
train_sales = M5.select_data(sales, category, train_dates)
train_prices = M5.select_data(prices, category, train_dates)
test_dates = M5.select_dates(dates, [2015], range(1, 13))
test_sales = M5.select_data(sales, category, test_dates)
test_prices = M5.select_data(prices, category, test_dates)

print('Setting Parameters ...')
groups = [[i for i in range(len(train_sales))]]

init_model_params = {
    'input_features': ['sale', 'price'],
    'output_feature': 'sale',
    'n_inputs': 2,
    'ckpt_path': ckpt_path
}

train_params = {
    'n_epochs': 100,
    'init_lr': 1e-4,
    'lr_decay_steps': 2
}

model_params_list = [
    ('input_features', [['sale', 'price'], ['sale']]),
    ('n_steps', list(range(3,10))),
    ('n_neurons', [100, 300, 500])
]

print('Runing Grid Search ...')

for idx, group in enumerate(groups):
    rnn = RNN.Model()
    rnn.set_model_params(init_model_params)
    rnn.set_training_params(train_params)
    group_train_sales = [train_sales[idx] for idx in group]
    group_test_sales = [test_sales[idx] for idx in group]
    group_train_prices = [train_prices[idx] for idx in group]
    group_test_prices = [test_prices[idx] for idx in group]
    train_datalist = {
        'sale': group_train_sales,
        'price': group_train_prices
    }
    test_datalist = {
        'sale': group_test_sales,
        'price': group_test_prices
    }
    if not path.exists(perform_path):
        makedirs(perform_path)
    grid_search_model_params(
        rnn,
        name,
        {},
        model_params_list,
        train_datalist,
        test_datalist,
        path.join(perform_path, 'results_' + str(idx) + '.csv'),
        model_path
    )
    print('Finished Group ' + str(idx))