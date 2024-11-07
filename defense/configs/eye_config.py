import albumentations as A
import cv2
from default_config import basic_cfg
import sys
sys.path.append('../')

cfg = basic_cfg

cfg.data_dir = '../data/EyePACS_AIROGS'
cfg.GAN_path = './net/GAN/pretrained_model/network-snapshot-003100_eye.pkl'

# train
cfg.train = True
cfg.eval = True

# per_client
cfg.client = [
    {'batch_size':4, 'lr':1e-2, 'GAN_lr': 1e-3, 'GAN_round': 5},                
    {'batch_size':4, 'lr':1e-2, 'GAN_lr': 1e-3, 'GAN_round': 5},                
    {'batch_size':4, 'lr':1e-2, 'GAN_lr': 1e-3, 'GAN_round': 5},                
    {'batch_size':4, 'lr':1e-2, 'GAN_lr': 1e-3, 'GAN_round': 5},                
    {'batch_size':8, 'lr':1e-2, 'GAN_lr': 2e-3, 'GAN_round': 5},                
    {'batch_size':8, 'lr':1e-2, 'GAN_lr': 2e-3, 'GAN_round': 5},                
    {'batch_size':8, 'lr':1e-2, 'GAN_lr': 2e-3, 'GAN_round': 5},                
    {'batch_size':8, 'lr':1e-2, 'GAN_lr': 2e-3, 'GAN_round': 5},                
    {'batch_size':1, 'lr':1e-2, 'GAN_lr': 1e-4, 'GAN_round': 5},                
]

# opt
cfg.optimizer = "sgd"
cfg.momentum = 0
cfg.weight_decay = 0
cfg.lr_schedule = "step"
cfg.lr_step = [40, 80]
cfg.epochs = 100
cfg.lr_server = 1
cfg.momentum_server = 0
cfg.optim_server = "sgd"    
cfg.lr_schedule_server = 'constant'

# shadow
cfg.shadow_train = True
cfg.shadow_final_epoch = 20
# pretrain z
cfg.z_iter = 500
cfg.z_lr = 1e-3
cfg.z_patience = 5
cfg.z_path = "../output_eye/client_z/"  # None
cfg.no_pretrained_z = False

# pseudo_train
cfg.pseudo_train_img = True

# fine-tune
cfg.no_pretrain = [f'L{str(i)}_' for i in range(7,15)] 

# GAN loss
cfg.shadow_grad_match = 1e-2
cfg.shadow_bn = 1e-2
cfg.shadow_total_var = 1e-5
cfg.shadow_img_norm = 1e2
cfg.shadow_rec = 1

# noise
cfg.noise_equ = True
cfg.noise_momentum = 0.1
cfg.sub_cam = True
cfg.perc_cam = 0.3
cfg.min_cam_alpha = 0.1
cfg.max_cam_alpha = 0.5
cfg.noise_rescale = 0.19    

# true update
cfg.ema_shadow = 0.5

# change noise according to decreasing length
cfg.noise_increase = True

# dataset
cfg.img_size = 224
cfg.val_size = 224

# val
cfg.eval_epochs = 1
cfg.start_eval_epoch = 0

# FL
cfg.local_round = 1
cfg.com_fre = 1
cfg.client_num = 9
cfg.save_grad = [0, cfg.epochs//4-1, cfg.epochs//2-1, cfg.epochs*3//4-1, cfg.epochs-1]

# model
cfg.output_dir = "../output_eye/"

# resume
cfg.resume = False
cfg.weights_path = None

# test
cfg.test_path = None

# transforms
cfg.norm_mean = (0.1168, 0.1869, 0.2992)
cfg.norm_std = (0.1493, 0.1901, 0.2815)

cfg.train_transforms =  A.Compose([
                A.ToFloat(max_value=255.0),
                A.Normalize(
                mean=cfg.norm_mean,
                std=cfg.norm_std,
                max_pixel_value=1.0,
                ),
                A.Resize(height=cfg.img_size, width=cfg.img_size, interpolation=cv2.INTER_CUBIC),
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