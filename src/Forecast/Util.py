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

def generate_data_matrix(data, n_steps, n_preds):
    m = len(data)
    data_mat = []
    for i in range(m):
        n = len(data[i])
        n_slides = n - n_steps - n_preds + 1
        mat = np.zeros((n_slides, n_steps + n_preds, 1))
        for j in range(n_slides):
            mat[j, :, 0] = data[i][j:j + n_steps + n_preds]
        data_mat.append(mat)
    return np.concatenate(data_mat, axis = 0)

def create_group_sets_with_datalist(train_datalist, test_datalist, groups, n_steps, n_preds):
    train_sets, val_sets = [], []
    for group in groups:
        train_mats, val_mats = [], []
        for train_data, test_data in zip(train_datalist, test_datalist):
            group_train_series = [train_data[i] for i in group]
            group_val_series = [train_data[i][-n_steps - n_preds + 1:] + test_data[i] for i in group]
            train_mat = generate_data_matrix(group_train_series, n_steps, n_preds)
            val_mat = generate_data_matrix(group_val_series, n_steps, n_preds)
            train_mats.append(train_mat)
            val_mats.append(val_mat)
        train_mat = np.concatenate(train_mats, axis = 2)
        val_mat = np.concatenate(val_mats, axis = 2)
        train_X = train_mat[:, :-n_preds,:]
        train_y = train_mat[:, n_preds:,:]
        val_X = val_mat[:, :-n_preds,:]
        val_y = val_mat[:, n_preds:,:]
        train_sets.append(Dataset(train_X, train_y))
        val_sets.append(Dataset(val_X, val_y))
    return train_sets, val_sets

def create_group_sets(train_data, test_data, groups, n_steps, n_preds):
    train_sets, val_sets = [], []
    for group in groups:
        group_train_series = [train_data[i] for i in group]
        group_val_series = [train_data[i][-n_steps - n_preds + 1:] + test_data[i] for i in group]
        train_mat = generate_data_matrix(group_train_series, n_steps, n_preds)
        val_mat = generate_data_matrix(group_val_series, n_steps, n_preds)
        train_X = train_mat[:, :-n_preds,:]
        train_y = train_mat[:, n_preds:,:]
        val_X = val_mat[:, :-n_preds,:]
        val_y = val_mat[:, n_preds:,:]
        train_sets.append(Dataset(train_X, train_y))
        val_sets.append(Dataset(val_X, val_y))
    return train_sets, val_sets    

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