import scipy.io
import numpy as np

def import_data_from_mat(mat_path, years, months):
    data_dict = scipy.io.loadmat(mat_path)
    if 'data' in data_dict:
        data = data_dict['data']
    else:
        data = data_dict['data_inv']
    year_mask = np.isin(data[:, 0], years)
    month_mask = np.isin(data[:, 1], months)
    selected_data = np.transpose(data[year_mask & month_mask, 3:])
    m, n = selected_data.shape
    ts = [[] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ts[i].append(selected_data[i,j])
    return ts