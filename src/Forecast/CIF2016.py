from .Util import get_data_from_txt

def import_data(data_path, head_len):
    data = get_data_from_txt(data_path, head_len)
    return data