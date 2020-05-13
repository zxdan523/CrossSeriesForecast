from Forecast.Util import *
from Forecast import EST, LinearRegression, SVR, RandomForest, MLP, RNN
from Forecast import M5
from Forecast.Grouping import *
import argparse
import numpy as np
from os import path, makedirs
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type = str,
    help = 'Model Type (EST, LR, SVR, RF, MLP, RNN)'
)
parser.add_argument(
    '--dept_id',
    type = str,
    help = 'department id (FOODS_1, FOODS_2, FOODS_3,\
    HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2)'
)
parser.add_argument(
    '--store_id',
    type = str,
    help = 'store id (CA_1, CA_2, CA_3, CA_4,\
    TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)'
)
parser.add_argument(
    '--training_years',
    type = int,
    nargs = '+',
    help = 'training years from 2011 - 2015'
)
parser.add_argument(
    '--test_year',
    type = int,
    help = 'test years from 2012 - 2016'
)
parser.add_argument(
    '--group_type',
    type = str,
    help = 'group type (no_group, global, vol, etc.)'
)
parser.add_argument(
    '--params_file',
    type = str,
    help = 'parameters file path'
)
parser.add_argument(
    '--dataset_path',
    type = str,
    default = '../data/m5-forecasting-accuracy',
    help = 'path of dataset'
)
parser.add_argument(
    '--base_path',
    type = str,
    default = '../data/M5',
    help = 'base path'
)

args = parser.parse_args()

print('Loading Data ...')
sales, prices = M5.import_weekly_sales_and_prices(args.dataset_path)
dates = M5.import_weekly_dates(args.dataset_path)

dept_id = args.dept_id
store_id = args.store_id
training_years = args.training_years
test_years = [args.test_year]
name = 'M5_' +\
    dept_id + \
    '_' + store_id + \
    '_' + args.model + \
    '_' + args.group_type + \
    '_' + str(args.test_year)


base_path = path.join(
    args.base_path,
    dept_id + '_' + store_id,
    args.model,
    str(args.test_year),
    args.group_type
)
perform_path = path.join(base_path, 'performance')
model_path = path.join(base_path, 'models')
ckpt_path = path.join(base_path, 'ckpts')
group_path = path.join(
    args.base_path,
    dept_id + '_' + store_id,
    'groups',
    args.group_type + '.txt'
)

category = {'dept_id': dept_id, 'store_id':store_id}
train_dates = M5.select_dates(dates, training_years, range(1, 13))
train_sales = M5.select_data(sales, category, train_dates)
train_prices = M5.select_data(prices, category, train_dates)
test_dates = M5.select_dates(dates, test_years, range(1, 13))
test_sales = M5.select_data(sales, category, test_dates)
test_prices = M5.select_data(prices, category, test_dates)

print('Setting Parameters ...')
groups = import_groups(group_path)

with open(args.params_file) as json_file:
    params = json.load(json_file)
    init_model_params = params['init_model_params']
    init_model_params['ckpt_path'] = ckpt_path
    training_params = params['training_params']
    model_params_list = params['model_params_list']
    
if args.model == 'EST':
    ML = EST
elif args.model == 'LR':
    ML = LinearRegression
elif args.model == 'SVR':
    ML = SVR
elif args.model == 'RF':
    ML = RandomForest
elif args.model == 'MLP':
    ML = MLP
elif args.model == 'RNN':
    ML = RNN
else:
    sys.exit('The model should be in (EST, LR, SVR, RF, MLP, RNN)')

print('Runing Grid Search ...')

for idx, group in enumerate(groups):
    model = ML.Model()
    model.set_model_params(init_model_params)
    model.set_training_params(training_params)
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
        model,
        name,
        {},
        model_params_list,
        train_datalist,
        test_datalist,
        path.join(perform_path, 'results_' + str(idx) + '.csv'),
        model_path
    )
    print('Finished Group ' + str(idx))