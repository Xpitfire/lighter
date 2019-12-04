import os
import wget
import torch
from PIL import Image
from transforms.test import Transform
from lighter.context import Context
from models.custom_conv import CustomConvModel
from lighter.utils.inet import download_and_extract_zip


def load_pretrained(model, path, device='cpu', model_state_dict='model_state_dict'):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt[model_state_dict]
    model.load_state_dict(state_dict)


def classify(model, image_file, target: str):
    transform = Transform()
    im = torch.unsqueeze(transform(Image.open(image_file)), dim=0)
    prediction = model(im)
    print('\nLooking {}: '.format(target), prediction.item())


if __name__ == '__main__':
    Context.create(device='cpu')
    model = CustomConvModel()
    pretrained_weights = 'runs/close-burro/deeply-secure-lizard/e-25_time-1575458055.645044.ckpt'
    load_pretrained(model, pretrained_weights)
    neg_image_url = "https://peopledotcom.files.wordpress.com/2018/03/10-things-i-hate-about-you-house-6.jpg"
    pos_image_url = "https://www.midlandsderm.com/wp-content/uploads/2019/04/Rachel-R.-Person-760x760.jpg"
    local_pos_image = "data/extract/186054-564091.jpg"
    local_neg_image = "data/extract/183830-266400.jpg"
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    neg_local_image_filename = wget.download(neg_image_url, out='tmp')
    pos_local_image_filename = wget.download(pos_image_url, out='tmp')

    classify(model, local_neg_image, 'neg from dataset')
    classify(model, local_pos_image, 'pos from dataset')
    classify(model, neg_local_image_filename, 'neg from url')
    classify(model, pos_local_image_filename, 'pos from url')
