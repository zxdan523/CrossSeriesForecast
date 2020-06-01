import numpy as np
from os import path, makedirs
from Forecast.Util import *
import numpy.linalg as LA

def group_by_sorted_features(features, n_groups):
    sorted_idx = np.argsort(features)
    batch_size = (len(features)//n_groups)
    i_group = 0
    groups = []
    group = []
    for i, idx in enumerate(sorted_idx):
        group.append(int(idx))
        if ((i + 1) % batch_size) == 0:
            groups.append(group)
            group = []
            i_group += 1
            if i_group == n_groups - 1:
                break
    for i in range(batch_size * (n_groups - 1), len(features)):
        group.append(int(sorted_idx[i]))
    groups.append(group)
    return groups

def save_groups(groups, save_file):
    base_path = '/'.join(path.split(save_file)[:-1])
    if not path.exists(base_path):
        makedirs(base_path)
    with open(save_file, 'w') as txt_file:
        for group in groups:
            txt_file.write(
                ','.join(
                    [str(idx) for idx in group]
                ) + '\n'
            )
        
def import_groups(group_file):
    groups = []
    with open(group_file) as txt_file:
        for str_group in txt_file:
            group = [
                int(idx)\
                for idx\
                in str_group.split('\n')[0].split(',')
            ]
            groups.append(group)
    return groups


def Euler_dist(ts1, ts2):
    return LA.norm(np.array(ts1) - np.array(ts2)) / len(ts1)

def ts_dist(ts1, ts2, dist_func = Euler_dist):
    count = 0
    dist = 0
    start = 0
    while start < len(ts1):
        end = start
        while end < len(ts1) and int(ts1[end]) and int(ts2[end]):
            end += 1
        if end > start:
            dist += dist_func(ts1[start:end], ts2[start:end])
            count +=1
        start = end + 1
    if count == 0:
        return Euler_dist(ts1, ts2)
    return dist / count

def get_dist_mat(ts, dist_func = Euler_dist):
    n = len(ts)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = ts_dist(ts[i], ts[j])
    return dist_mat
            
            