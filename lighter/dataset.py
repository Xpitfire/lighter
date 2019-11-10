import zipfile, os
from torch.utils.data import Dataset
from lighter.decorator import context
import urllib.request as request
import shutil


class BaseDataset(Dataset):
    @context
    def __init__(self):
        pass

    @staticmethod
    def download_zip(url, dest_path):
        tmp_path = 'tmp'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        data, _ = request.urlretrieve(url, filename=os.path.join(tmp_path, 'file.zip'))
        with open(data, 'rb') as file:
            zf = zipfile.ZipFile(file)
            zf.extractall(path=dest_path)
        shutil.rmtree(tmp_path)
