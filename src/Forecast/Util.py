import numpy as np
import csv

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

def fomalize_predictions_as_data(predictions, groups, data):
    results = [[0] * len(item) for item in data]
    for i, group in enumerate(groups):
        start = 0
        for idx in group:
            num_val = len(results[idx])
            results[idx] = predictions[i][start:start + num_val]
            start += num_val
    return results

def evaluate_NMSE(predictions, ys):
    NMSEs = []
    for pred, y in zip(predictions, ys):
        NMSEs.append(
            np.mean(
                np.power(np.array(pred) - np.array(y), 2) / np.mean(pred) / np.mean(y)
            )
        )
    pred = np.concatenate(predictions)
    y = np.concatenate(ys)
    NMSEs.append(
        np.mean(
            np.power(np.array(pred) - np.array(y), 2) / np.mean(pred) / np.mean(y)
        )
    )
    return NMSEs

def evaluate_SMAPE(predictions, ys):
    SMAPEs = []
    for pred, y in zip(predictions, ys):
        SMAPEs.append(
            np.mean(
                np.abs(np.array(pred) - np.array(y)) / (np.abs(pred) + np.abs(y)) * 2
            )
        )
    return SMAPEs