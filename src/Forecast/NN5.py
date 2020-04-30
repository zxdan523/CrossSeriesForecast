from .Util import get_data_from_txt

def import_data(data_path):
    data = get_data_from_txt(data_path, 0)
    m, n = len(data), len(data[0])
    data = [[data[i][j] for i in range(m)] for j in range(n)]
    return data