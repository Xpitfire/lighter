import zipfile, os
import urllib.request as request
import shutil


def download_and_extract_zip(url, dest_path):
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
