import os
import click
import json
import tempfile
import copy
import torch
import sys
sys.path.append('./net/GAN')
import dnnlib as dnnlib
import legacy as legacy
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

def load_pretrained_GAN(network_pkl):

    dnnlib.util.Logger(should_flush=True)

    args = dnnlib.EasyDict(num_gpus=1, network_pkl=network_pkl, verbose=True)

    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module
        # args.D = network_dict['D']

        # Print network summary.
        rank = 0
        device = torch.device('cuda', rank)
        G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
        # D = copy.deepcopy(args.D).eval().requires_grad_(False).to(device)
        # for key in G.state_dict().keys():   # D
        #     print(key)

        # if rank == 0 and args.verbose:
        #     z = torch.empty([1, G.z_dim], device=device)
        #     c = torch.empty([1, G.c_dim], device=device)
        #     misc.print_module_summary(G, [z, c])

    return G

