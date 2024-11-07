import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import argparse
import importlib
from utils.utils import *
from my_train.train_fed import main

if __name__ == '__main__':

    sys.path.append("configs")

    parser = argparse.ArgumentParser()
    available_config = ['chestXray_config', 'eye_config']
    parser.add_argument("-c", "--config", choices=available_config, default="eye_config", help="config filename")
    parser.add_argument('--log', type=str2bool, default='True', help='Whether to log')
    parser.add_argument('--seed', type = int, default=42, help = 'random seed')
    parser.add_argument('--gpu', type = int, default=1, help = 'gpu number')
    parser.add_argument('--server_gpu', type = int, default=0, help = 'gpu index of server')
    parser.add_argument('--exp_name', type = str, default='FedAvg', help = 'exp_name')

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.config = parser_args.config
    cfg.log = parser_args.log
    cfg.seed = parser_args.seed
    cfg.gpu = parser_args.gpu
    cfg.server_gpu = parser_args.server_gpu
    cfg.exp_name = parser_args.exp_name

    set_seed(cfg.seed)

    main(cfg)
    