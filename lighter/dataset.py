import zipfile, os

from torch.utils.data import Dataset
from lighter.decorator import transform
import urllib.request as request
import shutil


class BaseDataset(Dataset):
    """
    Dataset base class preparing the data.
    """
    @transform
    def __init__(self):
        pass

    @staticmethod
    def download_zip(url, dest_path):
        """
        Downloads and extracts a zip-file to the destination path.
        :param url: url to download zip file.
        :param dest_path: destination path to extract the files
        :return:
        """
        tmp_path = 'tmp'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        data, _ = request.urlretrieve(url, filename=os.path.join(tmp_path, 'file.zip'))
        with open(data, 'rb') as file:
            zf = zipfile.ZipFile(file)
            zf.extractall(path=dest_path)
        shutil.rmtree(tmp_path)
