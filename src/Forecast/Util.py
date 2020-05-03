from datetime import datetime
import numpy as np
import csv
from os import path, makedirs

class Dataset:
    def __init__(self, X = [], y = []):
        self.X = X
        self.y = y
    
# class M5:
#    def import_data(data_path):
        
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
    return np.mean(\
                np.power(np.array(pred) - np.array(y), 2) / np.mean(pred) / np.mean(y)\
            )

def eva_NMAE(pred, y):
    return np.mean(np.abs(np.array(pred) - np.array(y))) / np.mean(y)


def eva_R(pred, y):
    return 1 - np.sum(np.power(np.array(pred) - np.array(y),2)) /\
        np.sum(np.power(np.array(y) - np.mean(y),2))

def eva_SMAPE(pred, y):
    return np.mean(\
                np.abs(np.array(pred) - np.array(y)) / (np.abs(pred) + np.abs(y)) * 2\
            )
def eva(preds, ys, metric):
    results = []
    for pred, y in zip(preds, ys):
        results.append(metric(pred, y))
    return results

def save_to_file(content, header, file_path):
    file_mode = 'a'
    if not path.exists(file_path):
        file_mode = 'w'
    with open(file_path, file_mode) as save_file:
        if file_mode == 'w':
            save_file.write(','.join(header) + '\n')
        str_content = [str(content[k]) for k in header]
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
        now = datetime.now()
        new_model_params['name'] =\
            model_name +\
            '_' +\
            now.strftime("%Y-%m-%d-%H-%M-%S")
        
        model.set_model_params(new_model_params)
        
        train_set = model.create_set(train_datalist)
        val_datalist = {}
        for feature in train_datalist:
            val_datalist[feature] =\
                [train_item[-model.n_steps:] + test_item\
                 for train_item, test_item in zip(\
                                                  train_datalist[feature],\
                                                  test_datalist[feature]\
                                                 )\
                ]
        val_set = model.create_set(val_datalist)
        test_data = test_datalist[model.output_features[0]]
        
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
        print('tunning param ' + str(param_name) + ':' + str(param_value))
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
    