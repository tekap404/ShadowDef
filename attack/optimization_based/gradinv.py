import os, sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import argparse
from copy import deepcopy
import numpy as np
import torch
from fl_gradient_inversion import FLGradientInversion
from monai.networks.nets.torchvision_fc import TorchVisionFCModel
from util import set_seed, str2bool
import copy
import warnings
warnings.filterwarnings("ignore")

class GradInversionInverter():
    def __init__(
        self,
        args,
        bn_momentum: float = 0.1,
        save_fmt = ".png",
    ):
        self.args = args
        self.bn_momentum = bn_momentum
        self.save_fmt = save_fmt

    def run_inversion(
        self, args, updates, global_weights, bn_momentum=0.1, save_fmt=".png"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = TorchVisionFCModel(
            model_name='resnet18',
            num_classes=2,
            pretrained=False,
        )

        # get global weights                                                
        if "model" in global_weights:
            net.load_state_dict(global_weights["model"])
        else:
            net.load_state_dict(global_weights)

        # compute weight changes
        model_bn = deepcopy(net).cuda()
        new_state_dict = model_bn.state_dict()
        for n in updates.keys():
            val = updates[n]
            new_state_dict[n] += torch.tensor(
                val,
                dtype=new_state_dict[n].dtype,
                device=new_state_dict[n].device,
            )
        model_bn.load_state_dict(new_state_dict)
        n_bn_updated = 0
        global_state_dict = net.state_dict()

        # Compute full BN stats
        bn_stats = {}
        for param_name in updates:
            if "bn" in param_name or "batch" in param_name or "running" in param_name:
                bn_stats[param_name] = global_weights[param_name] + updates[param_name]     
        for n in bn_stats.keys():
            if "running" in n:
                xt = (bn_stats[n] - (1 - bn_momentum) * global_state_dict[n].numpy()) / bn_momentum     
                n_bn_updated += 1
                bn_stats[n] = xt

        # move weight updates and model to gpu
        net = net.to(device)
        grad_lst = []
        for name, _ in net.named_parameters():                                                 
            val = updates[name].cuda()
            grad_lst.append([name, val])

        # Compute inversion
        grad_inversion_engine = FLGradientInversion(
            network=net,
            grad_lst=grad_lst,
            bn_stats=bn_stats,
            model_bn=model_bn,
        )

        best_inputs, targets = grad_inversion_engine(args)

        return best_inputs, targets

    def __call__(self):

        global_model_weights = torch.load(self.args.global_model_path, map_location='cpu')
        # get updates
        weight_updates = torch.load(self.args.grad_path, map_location='cpu')

        # run inversion
        best_images, _ = self.run_inversion(
            args=self.args,
            updates=weight_updates,
            global_weights=global_model_weights,
            bn_momentum=self.bn_momentum,
            save_fmt=self.save_fmt,
        )

        return best_images

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad_inversion')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    available_config = ['chestXray_config', 'eye_config']
    parser.add_argument("-c", "--config", choices=available_config, default="chestXray_config", help="config filename")
    parser.add_argument('--root', default='./', type=str)
    parser.add_argument('--output_path', default='../output_attack/client8_prec0', type=str)    # 输出路径
    parser.add_argument('--global_model_path', default='../output_checkXray/FedAvg/model/global_model_epoch1.pth', type=str)
    parser.add_argument('--grad_path', default='../output_checkXray/FedAvg/client8_model/client8_grad_epoch1.pkl', type=str)
    parser.add_argument('--start_rand', type=str2bool, default='False')
    parser.add_argument('--init_target_rand', type=str2bool, default='False')

    parser.add_argument('--pretrained_size', nargs='+',         
                        type=int, default=[1, 3, 224, 224])
    parser.add_argument('--client_id', type=int, default=8)     
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--target_idx', type=int, default=0)
    parser.add_argument('--lr_local', type=float, default=1e-2)                      
    parser.add_argument('--local_optim', default='sgd', type=str)               
    parser.add_argument('--local_epoch', type=int, default=1)

    parser.add_argument('--criterion', default='BCEWithLogitsLoss', type=str)   
    parser.add_argument('--iterations', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=1e-1)               
    parser.add_argument('--tv_l2', type=float, default=1e-4)
    parser.add_argument('--tv_l1', type=float, default=0.)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--grad_l2', type=float, default=1e-3)
    parser.add_argument('--grad_cos', type=float, default=1e-5)
    parser.add_argument('--original_bn_l2', type=float, default=1e-1)
    parser.add_argument('--energy_l2', type=float, default=1e-1)

    # 保存
    parser.add_argument('--save_fre', type=int, default=500)

    parser.add_argument('--ViT', type=str2bool, default='False')

    args = parser.parse_args()
    args.batch_size = args.pretrained_size[0]
    args.channel = args.pretrained_size[1]
    args.resolution = args.pretrained_size[2]

    set_seed(args.seed)
    if args.config == 'chestXray_config':
        args.template_path ='../data/ChestX-ray14/mean_img.png'
    elif args.config == 'eye_config':
        args.template_path ='../data/EyePACS_AIROGS/mean_img.png'

    gradinv = GradInversionInverter(args)
    best_images = gradinv()