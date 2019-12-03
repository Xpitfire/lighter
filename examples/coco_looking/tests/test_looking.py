import os
import wget
import torch
from PIL import Image
from transforms.inception import Transform
from lighter.context import Context
from models.inception import InceptionNetFeatureExtractionModel
from lighter.utils.inet import download_and_extract_zip


def load_pretrained(model, path, device='cpu', model_state_dict='model_state_dict'):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt[model_state_dict]
    model.load_state_dict(state_dict)


if __name__ == '__main__':
    Context.create(device='cpu')
    model = InceptionNetFeatureExtractionModel()
    pretrained_dir = 'pretrained'
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    pretrained_weights = os.path.join(pretrained_dir, 'e-48_time-1575387566.827149.ckpt')
    if not os.path.exists(pretrained_weights):
        url = 'https://www.dinu.at/wp-content/uploads/2019/12/e-48_time-1575387566.827149.ckpt_.zip'
        download_and_extract_zip(url, pretrained_dir)
    load_pretrained(model, pretrained_weights)
    image_url = input('URL: ')
    local_image_filename = wget.download(image_url)
    transform = Transform()
    im = torch.unsqueeze(transform(Image.open(local_image_filename)), dim=0)
    prediction = model(im)
    print('\nLooking: ', prediction.item())
