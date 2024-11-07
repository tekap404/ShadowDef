from torch.utils import data as torchdata
import os
import pandas as pd
from PIL import Image
from  torch.utils.data import DataLoader
import numpy as np

def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image
    """
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()

class chestXray_dataset(torchdata.Dataset):
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
        img = single_channel_loader(path)
        img = np.expand_dims(np.array(img), -1).repeat(3, axis=-1)
        if self.transform:
            img = self.transform(image=img)['image']
        img = np.transpose(img, axes=(2,0,1))
        name = os.path.basename(path)
        return img, label, name

def get_partitioned_data(cfg, datadir):
    
    df = pd.read_csv(os.path.join(datadir, 'Chest_Xray_train.csv'))

    net_dataidx_map = {}
    for client in range(9):
        idx_list = df[df['client']==client].index.tolist()
        net_dataidx_map[client] = idx_list

    return net_dataidx_map, df

def get_chestXray_dataloader(cfg, df, dataidxs=None, client_id=None):

    df_train = pd.read_csv(os.path.join(cfg.data_dir, 'Chest_Xray_train.csv'))
    train_indices = dataidxs
    train_ds = chestXray_dataset(cfg, df=df_train, part_index=train_indices, train=True, transform=cfg.train_transforms)

    df_val = pd.read_csv(os.path.join(cfg.data_dir, 'Chest_Xray_val.csv'))
    val_indices = df_val[df_val['client']==client_id].index.tolist()
    val_ds = chestXray_dataset(cfg, df=df_val, part_index=val_indices, train=True, transform=cfg.val_transforms)
    
    df_test = pd.read_csv(os.path.join(cfg.data_dir, 'Chest_Xray_test.csv'))
    test_ds = chestXray_dataset(cfg, df=df_test, train=False, transform=cfg.test_transforms)

    train_dl = DataLoader(dataset=train_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_dl = DataLoader(dataset=val_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    test_dl = DataLoader(dataset=test_ds, batch_size=cfg.client[client_id]['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    return train_dl, val_dl, test_dl