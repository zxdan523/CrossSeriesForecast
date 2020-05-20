import os

model = 'RNN'
dept_id = 'FOODS_1'
store_id = 'CA_1'
training_years = '2012 2013 2014'
test_year = '2015'
group_type = 'global'
params_file = './RNN_params.json'
log_path = '../log/M5_' +\
    model + '_' +\
    dept_id + '_' +\
    store_id + '_' +\
    test_year + '_' +\
    group_type + '.txt'

os.system(
    'python ' +\
    'Run_M5.py ' +\
    '--model ' + model + ' ' +\
    '--dept_id ' + dept_id + ' ' +\
    '--store_id ' + store_id + ' ' +\
    '--training_years ' + training_years + ' ' +\
    '--test_year ' + test_year + ' ' +\
    '--group_type ' + group_type + ' ' +\
    '--params_file ' + params_file
)