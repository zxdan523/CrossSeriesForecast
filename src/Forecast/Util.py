from datetime import datetime
import numpy as np
import csv
import heapq
from os import path, makedirs, listdir, remove

available_metrics = {
    'R',
    'avg_R',
    'std_R',
    'NMSE',
    'avg_NMSE',
    'std_NMSE',
    'NMAE',
    'avg_NMAE',
    'std_NMAE',
    'SMAPE',
    'avg_SMAPE',
    'std_SMAPE'
}

larger_better_metrics = {'R', 'avg_R'}

def get_compare_func(metric_name):
    if metric_name in {'R', 'avg_R'}:
        compare_func = lambda x,y: x > y
    elif metric_name in {
        'NMSE',
        'avg_NMSE',
        'std_NMSE',
        'NMAE',
        'avg_NMAE',
        'std_NMAE',
        'std_R',
        'SMAPE',
        'avg_SMAPE',
        'std_SMAPE'
    }:
        compare_func = lambda x,y: x < y
    return compare_func

class Dataset:
    def __init__(self, X = [], y = []):
        self.X = X
        self.y = y

def get_best_preds(
    model,
    train_datalist,
    test_datalist,
    best_perform,
    metric,
    base_path
):
    best_model_name = best_perform[metric]['name']
    model.load_model_params(
        path.join(base_path,
              best_model_name + '.json'
             )
    )
    val_datalist = {}
    for feature in train_datalist:
        val_datalist[feature] = model.get_val_data(
            train_datalist[feature],
            test_datalist[feature]
        )
    val_set = model.create_set(val_datalist)
    preds = model.get_preds(val_set.X)
    
    return formalize_prediction_as_data(preds, test_datalist[model.output_feature])
        
def save_json_file(
    data,
    file_path
):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
    print('File is saved to ' + file_path)
def clear_model_and_ckpt_files(
    base_path,
    groups,
    top = 10
):
    perform_path = path.join(base_path, 'performance')
    model_path = path.join(base_path, 'models')
    ckpt_path = path.join(base_path, 'ckpts')
    if not path.exists(ckpt_path):
        makedirs(ckpt_path)
    n_groups = len(groups)
    for i in range(n_groups):
        result_path = path.join(perform_path, 'results_' + str(i) + '.csv')
        keep_models = set()
        opt_metric_heaps = {metric: [] for metric in available_metrics}
        with open(result_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_name = row['name']
                for metric_name in available_metrics:
                    if metric_name not in row:
                        continue
                    metric = float(row[metric_name])
                    if metric_name in larger_better_metrics:
                        value = (metric, model_name)
                    else:
                        value = (-1 * metric, model_name)
                    opt_heap = opt_metric_heaps[metric_name]
                    if len(opt_heap) < top:
                        heapq.heappush(opt_heap, value)
                    else:
                        heapq.heappushpop(opt_heap, value)
        for metric_name in available_metrics:
            for _, model_name in opt_metric_heaps[metric_name]:
                keep_models.add(model_name)
    count = 0
    for ckpt_file in listdir(ckpt_path):
        model_name = ckpt_file.split('.')[0]
        if not model_name in keep_models:
            remove(path.join(ckpt_path, ckpt_file))
            count += 1
    print('Delete {0} ckpt files in {1}'.format(count, ckpt_path))
    
    count = 0
    for model_file in listdir(model_path):
        model_name = model_file.split('.')[0]
        if not model_name in keep_models:
            remove(path.join(model_path, model_file))
            count += 1
    print('Delete {0} model files in {1}'.format(count, model_path))
    
def collect_best_model_preds(
    model,
    groups,
    train_datalist,
    test_datalist,
    base_path,
    metric_name
):
    perform_path = path.join(base_path, 'performance')
    model_path = path.join(base_path, 'models')
    n_groups = len(groups)
    compare_func = get_compare_func(metric_name)
    total_num = 0
    for group in groups:
        total_num += len(group)
    preds = [[] for _ in range(total_num)]
    for i in range(n_groups):
        result_path = path.join(perform_path, 'results_' + str(i) + '.csv')
        opt_model_name = None
        opt_metric = None
        with open(result_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_name = row['name']
                metric = float(row[metric_name])
                if opt_metric is None or compare_func(
                    metric,
                    opt_metric):
                    opt_metric = metric
                    opt_model_name = model_name
        model.load_model_params(
            path.join(model_path, opt_model_name + '.json')
        )
        group_train_datalist = {
            feature: [ 
                train_datalist[feature][idx]
                for idx in groups[i]
            ] for feature in train_datalist
        }
        group_test_datalist = {
            feature: [ 
                test_datalist[feature][idx]
                for idx in groups[i]
            ] for feature in test_datalist
        }
        group_val_datalist = {
            feature: model.get_val_data(
                    group_train_datalist[feature],
                    group_test_datalist[feature]
                ) for feature in train_datalist
        }
        val_set = model.create_set(group_val_datalist)
        group_preds = model.get_preds(val_set.X)
        start = 0
        for idx in groups[i]:
            item_size = len(test_datalist[model.output_feature][idx])
            preds[idx] = group_preds[
                start: start + item_size
            ]
            start += item_size
    return preds
    
def get_datalist_mean(datalist, include_zeros = False):
    averages = []
    if include_zeros:
        averages = [np.mean(data) for data in datalist]
        return averages
    for data in datalist:
        if np.mean(np.abs(data)) == 0:
            averages.append(0)
        else:
            averages.append(
                np.mean(
                    [item for item in data if item != 0]
                )
            )
    return averages
def get_datalist_std(datalist, include_zeros = False):
    stds = []
    if include_zeros:
        stds = [np.std(data) for data in datalist]
        return stds
    for data in datalist:
        if np.mean(np.abs(data)) == 0:
            stds.append(0)
        else:
            stds.append(
                np.std(
                    [item for item in data if item != 0]
                )
            )
    return stds
def get_datalist_coef_var(datalist, include_zeros = False):
    coef_vars = []
    if include_zeros:
        for data in datalist:
            data_mean = np.mean(data)
            data_std = np.std(data)
            if data_mean == 0:
                coef_vars.append("nan")
            else:
                coef_vars.append(data_std / data_mean)
        return coef_vars
    for data in datalist:
        if np.mean(np.abs(data)) == 0:
            coef_vars.append("nan")
        else:
            data_mean = np.mean(
                [item for item in data if item != 0]
            )
            data_std = np.std(
                [item for item in data if item != 0]
            )
            if data_mean == 0:
                coef_vars.append("nan")
            else:
                coef_vars.append(data_std / data_mean)
    return coef_vars
def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")
        
def get_data_from_csv(filename):
    data = []
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

def get_data_from_txt(filename, head_len):
    data = []
    def trans(v):
        if v in {'NA','NA\n'}:
            return 0
        return float(v)
    with open(filename) as txt_file:
        for line in txt_file:
            info = line.split(';')
            item = [trans(info[i]) for i in range(head_len,len(info))]
            data.append(item)
    return data   

def formalize_prediction_as_data(prediction, data):
    preds = [[] for _ in range(len(data))]
    start = 0
    for i, item in enumerate(data):
        preds[i] = prediction[start:start + len(item)]
        start += len(item)
    return preds

def fomalize_predictions_as_data(predictions, groups, data):
    results = [[0] * len(item) for item in data]
    for i, group in enumerate(groups):
        start = 0
        for idx in group:
            num_val = len(results[idx])
            results[idx] = predictions[i][start:start + num_val]
            start += num_val
    return results

def eva_NMSE(pred, y):
    pred = np.array(pred, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    return np.mean(\
                np.power(pred - y, 2) /\
                   np.abs(np.mean(pred) + 1e-6) /\
                   np.abs(np.mean(y) + 1e-6)\
            )

def eva_NMAE(pred, y):
    pred = np.array(pred, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    return np.mean(np.abs(pred - y)) / np.abs(np.mean(y) + 1e-6)


def eva_R(pred, y):
    pred = np.array(pred, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    return 1 - np.sum(np.power(pred - y,2)) /\
        (np.sum(np.power(y - np.mean(y),2)) + 1e-6)

def eva_SMAPE(pred, y):
    pred = np.array(pred, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    return np.mean(\
                np.abs(pred - y) /\
                   (np.abs(pred) + np.abs(y) + 1e-6) * 2\
            )
def eva_bias(pred, y):
    pred = np.array(pred, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    return np.mean(pred - y) / np.abs(np.mean(y) + 1e-6)

def eva(preds, ys, metric):
    results = []
    for pred, y in zip(preds, ys):
        if sum(y) == 0:
            continue
        results.append(metric(pred, y))
    return results

def save_to_file(content, header, file_path):
    file_mode = 'a'
    if not path.exists(file_path):
        file_mode = 'w'
    with open(file_path, file_mode) as save_file:
        if file_mode == 'w':
            save_file.write(','.join(header) + '\n')
        str_content = [str(content[k]).replace(',', ' ') for k in header]
        save_file.write(','.join(str_content) + '\n')
        
        
        
def grid_search_model_params(
    model,
    model_name,
    model_params,
    model_params_list,
    train_datalist,
    test_datalist,
    file_path,
    model_param_path
):
    if len(model_params_list) == 0:
        new_model_params = model_params.copy()
        print('Tunning Parameters:')
        for k in new_model_params:
            print(k + ':' + str(new_model_params[k]))
        now = datetime.now()
        new_model_params['name'] =\
            model_name +\
            '_' +\
            get_time()
        if new_model_params['name'] != model.name.split('+')[0]:
            new_model_params['name'] += '+0'
        else:
            new_model_params['name'] +=\
                '+' + str(int(model.name.split('+')[1]) + 1)
        model.set_model_params(new_model_params)
        
        train_set = model.create_set(train_datalist)
        val_datalist = {}
        for feature in train_datalist:
            val_datalist[feature] = model.get_val_data(\
                                                 train_datalist[feature],\
                                                 test_datalist[feature]\
                                                )
        val_set = model.create_set(val_datalist)
        test_data = test_datalist[model.output_feature]
        
        model.train(train_set, val_set)
        
        pred = model.get_preds(val_set.X)
        y = np.concatenate(test_data)
        
        results = {k:v for k,v in new_model_params.items()}
        results['R'] = eva_R(pred, y)
        results['NMSE'] = eva_NMSE(pred, y)
        results['NMAE'] = eva_NMAE(pred, y)
        results['SMAPE'] = eva_SMAPE(pred, y)
        
        preds = formalize_prediction_as_data(pred, test_data)
        Rs = eva(preds, test_data, eva_R)
        NMSEs = eva(preds, test_data, eva_NMSE)
        NMAEs = eva(preds, test_data, eva_NMAE)
        SMAPEs = eva(preds, test_data, eva_SMAPE)
        results['avg_R'] = np.mean(Rs)
        results['std_R'] = np.std(Rs)
        results['avg_NMSE'] = np.mean(NMSEs)
        results['std_NMSE'] = np.std(NMSEs)
        results['avg_NMAE'] = np.mean(NMAEs)
        results['std_NMAE'] = np.std(NMAEs)
        results['avg_SMAPE'] = np.mean(SMAPEs)
        results['std_SMAPE'] = np.std(SMAPEs)
        header = ['name']
        for name in new_model_params:
            if name not in header:
                header.append(name)
        for metric in {'R', 'NMSE', 'NMAE', 'SMAPE'}:
            header.append(metric)
            for stat in {'avg', 'std'}:
                header.append(stat + '_' + metric)
        
        save_to_file(results, header, file_path)
        model.save_model_params(model_param_path)
        return
    
    param_name, param_values = model_params_list[0]
    for param_value in param_values:
        new_model_params = model_params.copy()
        new_model_params[param_name] = param_value
        grid_search_model_params(
            model,
            model_name,
            new_model_params,
            model_params_list[1:],
            train_datalist,
            test_datalist,
            file_path,
            model_param_path
        )
        
def grid_search_model_params_v2(
    model,
    model_name,
    model_params,
    model_params_list,
    train_datalist,
    test_datalist,
    file_path,
    model_param_path,
    best_perform
):
    def better_than(metric, target, refer):
        if metric in {'R', 'avg_R'}:
            return target > refer
        return target < refer
    if len(model_params_list) == 0:
        new_model_params = model_params.copy()
        print('Tunning Parameters:')
        for k in new_model_params:
            print(k + ':' + str(new_model_params[k]))
        now = datetime.now()
        new_model_params['name'] =\
            model_name +\
            '_' +\
            get_time()
        if new_model_params['name'] != model.name.split('+')[0]:
            new_model_params['name'] += '+0'
        else:
            new_model_params['name'] +=\
                '+' + str(int(model.name.split('+')[1]) + 1)
        model.set_model_params(new_model_params)
        
        train_set = model.create_set(train_datalist)
        val_datalist = {}
        for feature in train_datalist:
            val_datalist[feature] = model.get_val_data(\
                                                 train_datalist[feature],\
                                                 test_datalist[feature]\
                                                )
        val_set = model.create_set(val_datalist)
        test_data = test_datalist[model.output_feature]
        
        model.train(train_set, val_set)
        
        pred = model.get_preds(val_set.X)
        y = np.concatenate(test_data)
        
        results = {k:v for k,v in new_model_params.items()}
        results['R'] = eva_R(pred, y)
        results['NMSE'] = eva_NMSE(pred, y)
        results['NMAE'] = eva_NMAE(pred, y)
        results['SMAPE'] = eva_SMAPE(pred, y)
        
        preds = formalize_prediction_as_data(pred, test_data)
        Rs = eva(preds, test_data, eva_R)
        NMSEs = eva(preds, test_data, eva_NMSE)
        NMAEs = eva(preds, test_data, eva_NMAE)
        SMAPEs = eva(preds, test_data, eva_SMAPE)
        results['avg_R'] = np.mean(Rs)
        results['std_R'] = np.std(Rs)
        results['avg_NMSE'] = np.mean(NMSEs)
        results['std_NMSE'] = np.std(NMSEs)
        results['avg_NMAE'] = np.mean(NMAEs)
        results['std_NMAE'] = np.std(NMAEs)
        results['avg_SMAPE'] = np.mean(SMAPEs)
        results['std_SMAPE'] = np.std(SMAPEs)
        header = ['name']
        for name in new_model_params:
            if name not in header:
                header.append(name)
        for metric in {'R', 'NMSE', 'NMAE', 'SMAPE'}:
            header.append(metric)
            for stat in {'avg', 'std'}:
                header.append(stat + '_' + metric)
                
        save_to_file(results, header, file_path)
        model.save_model_params(model_param_path)
        
        for metric in {'R',
                       'avg_R',
                       'NMSE',
                       'avg_NMSE',
                       'NMAE',
                       'avg_NMAE',
                       'SMAPE',
                       'avg_SMAPE'
                      }:
            if (metric in best_perform) and\
                (not better_than(
                    metric,
                    results[metric],
                    best_perform[metric]['value']
                    )
                ):
                continue
            best_perform[metric] = {}
            best_perform[metric]['value'] = results[metric]
            best_perform[metric]['name'] = new_model_params['name']

        return
    
    param_name, param_values = model_params_list[0]
    for param_value in param_values:
        new_model_params = model_params.copy()
        new_model_params[param_name] = param_value
        grid_search_model_params_v2(
            model,
            model_name,
            new_model_params,
            model_params_list[1:],
            train_datalist,
            test_datalist,
            file_path,
            model_param_path,
            best_perform
        )