import torch
import numpy as np
import random
import os
from torch.nn import init
from .loss import *
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

def str2bool(str):
	return True if str.lower() == 'true' else False

def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def renorm_resize(cfg, img, drange):
    lo, hi = drange
    img = torch.clamp(img, lo, hi)
    img = (img - lo) * (255 / (hi - lo))
    img = torch.clamp(img, 0, 255)

    norm = torch.Tensor([cfg.norm_mean]).repeat(img.shape[0], 1).unsqueeze(-1).unsqueeze(-1).to(img.device)
    std = torch.Tensor([cfg.norm_std]).repeat(img.shape[0], 1).unsqueeze(-1).unsqueeze(-1).to(img.device)
    img = (img - norm * 255) / (std * 255)

    img = F.interpolate(img, (cfg.img_size, cfg.img_size))
    
    return img

class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and
    compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, bn_stats=None, name=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.bn_stats = bn_stats
        self.name = name
        self.r_feature = None
        self.mean = None
        self.var = None

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)                                                        
        )

        var_feature = torch.norm(
            torch.tensor(
                self.bn_stats[self.name + ".running_var"], device=input[0].device
            ) - var,
            2,
        )
        mean_feature = torch.norm(
            torch.tensor(
                self.bn_stats[self.name + ".running_mean"], device=input[0].device
            ) - mean,
            2,
        )

        rescale = 1.0
        self.r_feature = mean_feature + rescale * var_feature                               
        self.mean = mean
        self.var = var

    def close(self):
        self.hook.remove()

def get_target_name(k_list, layer_new):
    full_name = []
    for k in k_list:
        for i in layer_new:
            if i in k:
                full_name.append(k)
    return full_name

def load_partial_model(new_model, pretrained_model, layer_new, cid):

    all_name = [k for k in pretrained_model.state_dict().keys()]
    target_name = get_target_name(all_name, layer_new)
    rest_name = set(all_name) - set(target_name)

    for k in target_name:
        if cid == 0:
            print(f'layer {k}: new')
    for k in rest_name:
        if cid == 0:
            print(f'layer {k}: pretrained')
        new_model.state_dict()[k].copy_(pretrained_model.state_dict()[k])

    return new_model

def set_model_grad(model, layer_new):

    for name, child in model.named_children():
        all_name = [k for k, v in child.named_parameters()]
        target_name = get_target_name(all_name, layer_new)
        rest_name = set(all_name) - set(target_name)

        for k in target_name:
            each_module = next((child for child_name, child in child.named_parameters() if k in child_name), None)
            each_module.requires_grad = True
        for k in rest_name:
            each_module = next((child for child_name, child in child.named_parameters() if k in child_name), None)
            each_module.requires_grad = False
    return model

def weighted_softmax(input_tensor, temperature):
    exp_values = torch.exp(input_tensor / temperature)
    sum_exp_values = torch.sum(exp_values)
    softmax_output = exp_values / sum_exp_values
    return softmax_output