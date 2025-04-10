import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True#False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

def str2bool(v):
    return v.lower() in ('true', '1') 

def to_img(x, dm, ds):
    '''
    x.shape = (n_imgs, n_channels, H, W)
    '''
    x = x.detach().clone().cpu()
    ds = ds.cpu()
    dm = dm.cpu()

    imgs = []
    for img in x:
        img = img * ds + dm
        img = img.clamp(0, 1)
        imgs.append(img)

    imgs = torch.stack(imgs)

    return imgs

def plt_imgs(imgs, y, n_channels, title=''):

    n_imgs = len(imgs)

    tt = transforms.ToPILImage()

    if n_imgs <= 8:
        plt.figure(figsize=(40, 24))
    else:
        plt.figure(figsize=(40, n_imgs))

    i = 0
    for img in imgs:
        plt.subplot(n_imgs*2 // 8 + 1, 8, i+1)
        if n_channels == 1:
            plt.imshow(tt(img), cmap='gray')
        else:
            plt.imshow(tt(img))
        plt.title(f'{title}:{y[i].item()}', fontsize=30)
        plt.axis('off')
        i += 1

def renorm_resize(args, img, drange):
    lo, hi = drange
    img = torch.clamp(img, lo, hi)
    img = (img - lo) * (255 / (hi - lo))
    img = torch.clamp(img, 0, 255)

    norm = torch.Tensor([args.norm_mean]).repeat(img.shape[0], 1).unsqueeze(-1).unsqueeze(-1).to(img.device)
    std = torch.Tensor([args.norm_std]).repeat(img.shape[0], 1).unsqueeze(-1).unsqueeze(-1).to(img.device)
    img = (img - norm * 255) / (std * 255)

    img = F.interpolate(img, (args.pretrained_size[-2], args.pretrained_size[-1]))
    
    return img