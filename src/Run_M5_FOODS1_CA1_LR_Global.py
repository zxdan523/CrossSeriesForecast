from Forecast.Util import *
from Forecast import LinearRegression
from Forecast import M5
import numpy as np
from os import path, makedirs

print('Loading Data ...')
sales, prices = M5.import_weekly_sales_and_prices('../data/m5-forecasting-accuracy')
dates = M5.import_weekly_dates('../data/m5-forecasting-accuracy')

dept_id = 'FOODS_1'
store_id = 'CA_1'
training_years = range(2012, 2015)
test_years = [2015]
name = 'M5_' + dept_id + '_' + store_id + '_LR_global'

base_path = '../data/M5/' + dept_id + '_' + store_id + '/LR/global'
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

model_params_list = [
    ('intercept', [False, True]),
    ('normalize', [False, True]),
    (
        'n_steps',
        [
            {'sale': s_steps, 'price': p_steps}\
            for s_steps in range(1,10)\
            for p_steps in range(10)
        ]
    )
]

print('Runing Grid Search ...')

for idx, group in enumerate(groups):
    lr = LinearRegression.Model()
    lr.set_model_params(init_model_params)
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
        lr,
        name,
        {},
        model_params_list,
        train_datalist,
        test_datalist,
        path.join(perform_path, 'results_' + str(idx) + '.csv'),
        model_path
    )
    print('Finished Group ' + str(idx))