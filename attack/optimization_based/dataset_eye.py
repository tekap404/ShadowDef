import torch
from tqdm import tqdm
from torch.utils import data as torchdata
import os
import pandas as pd
from PIL import Image
from  torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Optional, Tuple, Union
from random import random
from types import SimpleNamespace
import albumentations as A
import cv2
# from util import to_img, plt_imgs
import matplotlib.pyplot as plt

def img_loader(filename):
    """Converts `filename` to a grayscale PIL Image
    """
    with open(filename, "rb") as f:
        img = Image.open(f)
        return img.copy()

class eye_dataset(torchdata.Dataset):
    def __init__(
        self, cfg, df, part_index=None, train=False, transform=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.train = train
        if train == True:
            self.df = df.loc[part_index]
        else:
            self.df = df

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row["label"]
        path = row["img_path"]
        img = img_loader(path)
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img, axes=(2,0,1))
        name = os.path.basename(path)
        return img, label

def get_eye_dataloader(cfg, df, dataidxs=None, client_id=None):

    df_train = pd.read_csv(os.path.join(cfg.data_dir, 'eye_train.csv'))
    train_indices = dataidxs
    train_ds = eye_dataset(cfg, df=df_train, part_index=train_indices, train=True, transform=cfg.train_transforms)

    df_val = pd.read_csv(os.path.join(cfg.data_dir, 'eye_val.csv'))
    val_indices = df_val[df_val['client']==client_id].index.tolist()
    val_ds = eye_dataset(cfg, df=df_val, part_index=val_indices, train=True, transform=cfg.val_transforms)
    
    df_test = pd.read_csv(os.path.join(cfg.data_dir, 'eye_test.csv'))
    test_ds = eye_dataset(cfg, df=df_test, train=False, transform=cfg.test_transforms)

    train_dl = DataLoader(dataset=train_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_dl = DataLoader(dataset=val_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    test_dl = DataLoader(dataset=test_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    return train_dl, val_dl, test_dl

class eye_all_dataset(torchdata.Dataset):
    def __init__(
        self, cfg, df, transform=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        client_idx = row["client"]
        path = row["img_path"]
        img = img_loader(path)
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img, axes=(2,0,1))
        return img, client_idx

def load_prior(args):

    img = img_loader(args.template_path)
    img = np.array(img)
    print(img.shape)
    cfg = SimpleNamespace(**{})
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])
    cfg.norm_mean = (0.1168, 0.1869, 0.2992)
    cfg.norm_std = (0.1493, 0.1901, 0.2815)

    cfg.train_transforms =  A.Compose([
                        A.ToFloat(max_value=255.0),
                        A.Normalize(
                            mean=cfg.norm_mean,
                            std=cfg.norm_std,
                            max_pixel_value=1.0,
                        ),
                        A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
        ], p=1.0)

    template = cfg.train_transforms(image=img)['image']
    template = torch.tensor(np.transpose(template, axes=(2,0,1)))
    return template

def load_rec_imgs(args):

    rec_x = []
    cfg = SimpleNamespace(**{})
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])

    cfg.train_transforms =  A.Compose([
                        A.ToFloat(max_value=255.0),
                        A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
        ], p=1.0)
    for bs in range(args.pretrained_size[0]):
        img = cv2.imread(os.path.join(args.output_path, f'recon_{str(bs)}.png'))
        print(os.path.join(args.output_path, f'recon_{str(bs)}.png'), img.shape)
        img = cfg.train_transforms(image=img)['image']
        img = torch.tensor(np.transpose(img, axes=(2,0,1)))
        rec_x.append(img)
    rec_x = torch.stack(rec_x, dim=0)
    
    return rec_x

def load_nonorm_prior(args):

    img = img_loader(args.template_path)
    img = np.array(img)
    cfg = SimpleNamespace(**{})
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])

    cfg.train_transforms =  A.Compose([
                        A.ToFloat(max_value=255.0),
                        A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
        ], p=1.0)

    template = cfg.train_transforms(image=img)['image']
    template = torch.tensor(np.transpose(template, axes=(2,0,1)))
    return template

def get_partitioned_data_eye(cfg, datadir):
    
    df = pd.read_csv(os.path.join(datadir, 'eye_train.csv'))

    net_dataidx_map = {}
    for client in range(9):
        idx_list = df[df['client']==client].index.tolist()
        net_dataidx_map[client] = idx_list

    return net_dataidx_map, df

def load_dataloader(args):

    cfg = SimpleNamespace(**{})
    cfg.data_dir = '../data/EyePACS_AIROGS'
    cfg.fold = args.fold_idx
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])

    cfg.train_transforms =  A.Compose([
                A.ToFloat(max_value=255.0),
                A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
            ], p=1.0)

    net_dataidx_map, df = get_partitioned_data_eye(cfg, datadir=cfg.data_dir)
    df_train = pd.read_csv(os.path.join(cfg.data_dir, 'eye_train.csv'))
    train_ds = eye_dataset(cfg, df=df_train, part_index=net_dataidx_map[args.client_id], train=True, transform=cfg.train_transforms)
    train_dl = DataLoader(dataset=train_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    return train_dl

def load_all_dataloader(args):

    cfg = SimpleNamespace(**{})
    cfg.data_dir = '../data/EyePACS_AIROGS'
    cfg.fold = args.fold_idx
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])

    cfg.train_transforms =  A.Compose([
                A.ToFloat(max_value=255.0),
                A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
            ], p=1.0)

    df_train = pd.read_csv(os.path.join(cfg.data_dir, 'eye_train.csv'))
    df_val = pd.read_csv(os.path.join(cfg.data_dir, 'eye_val.csv'))
    df_val['client'] = -1  
    df = pd.concat([df_train, df_val])
    all_ds = eye_all_dataset(cfg, df=df, transform=cfg.train_transforms)
    all_dl = DataLoader(dataset=all_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    return all_dl


def get_target_samples(args):

    cfg = SimpleNamespace(**{})
    cfg.data_dir = '../data/EyePACS_AIROGS'
    cfg.fold = args.fold_idx
    cfg.client = [
        {'batch_size':4, 'lr':1e-2},
        {'batch_size':4, 'lr':1e-2},
        {'batch_size':4, 'lr':1e-2},
        {'batch_size':4, 'lr':1e-2},
        {'batch_size':8, 'lr':1e-2},
        {'batch_size':8, 'lr':1e-2},
        {'batch_size':8, 'lr':1e-2},
        {'batch_size':8, 'lr':1e-2},
        {'batch_size':1, 'lr':1e-2},
    ]
    cfg.batch_size = cfg.client[args.client_id]['batch_size']
    cfg.img_size = int(args.pretrained_size[-2])
    cfg.val_size = int(args.pretrained_size[-2])
    cfg.norm_mean = (0.1168, 0.1869, 0.2992)
    cfg.norm_std = (0.1493, 0.1901, 0.2815)

    cfg.train_transforms =  A.Compose([
                    A.ToFloat(max_value=255.0),
                    A.Normalize(
                    mean=cfg.norm_mean,
                    std=cfg.norm_std,
                    max_pixel_value=1.0,
                    ),
                    A.Resize(height=cfg.val_size, width=cfg.val_size, interpolation=cv2.INTER_CUBIC),
            ], p=1.0)

    cfg.val_transforms = A.Compose([
            A.ToFloat(max_value=255.0),
            A.Normalize(
                mean=cfg.norm_mean,
                std=cfg.norm_std,
                max_pixel_value=1.0,
            ),
            A.Resize(cfg.val_size, cfg.val_size, interpolation=cv2.INTER_CUBIC),
        ], p=1.0)

    cfg.test_transforms = A.Compose([
            A.ToFloat(max_value=255.0),
            A.Normalize(
                mean=cfg.norm_mean,
                std=cfg.norm_std,
                max_pixel_value=1.0,
            ),
            A.Resize(cfg.val_size, cfg.val_size, interpolation=cv2.INTER_CUBIC),
        ], p=1.0)
    
    net_dataidx_map, df = get_partitioned_data_eye(cfg, datadir=cfg.data_dir)
    loader, _, _ = get_eye_dataloader(cfg, df, net_dataidx_map[args.client_id], args.client_id)

    # load target samples
    for i in range(args.target_idx + 1):
        x_true, y_true = next(iter(loader))

    dm = torch.as_tensor(cfg.norm_mean).view(args.channel, 1, 1)
    ds = torch.as_tensor(cfg.norm_std).view(args.channel, 1, 1)

    # # save true image
    # true_imgs = to_img(x_true, dm, ds)
    # plt_imgs(true_imgs, y_true, args.channel, title='T')
    # plt.savefig(os.path.join(args.output_path, 'true.png'))
    # plt.close()

    y_true_tmp = torch.zeros((args.batch_size, 2))
    for bs in range(len(y_true)):
        if y_true[bs] == 0:
            y_true_tmp[bs, 0] = 1
        else:
            y_true_tmp[bs, 1] = 1
    del y_true
    y_true = y_true_tmp
    
    return x_true, y_true, dm, ds