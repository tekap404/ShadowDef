# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import collections
import json
import logging
import os
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm
import copy
import torch.nn.functional as F

class FLGradientInversion(object):
    def __init__(
        self,
        network,
        grad_lst,
        bn_stats,
        model_bn,
    ):
        self.network = network
        self.net0 = network
        self.bn_stats = bn_stats
        self.model_bn = model_bn
        self.loss_r_feature_layers = []
        self.grad_lst = grad_lst

    def __call__(self, args):

        if args.config == 'chestXray_config':
            from dataset_chestXray import load_prior, get_target_samples
        elif args.config == 'eye_config':
            from dataset_eye import load_prior, get_target_samples

        self.save_path = args.output_path
        save_every = args.save_fre
        if save_every > 0:
            self.create_folder(self.save_path)

        if args.criterion == "BCEWithLogitsLoss":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.criterion == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = self.network
        local_rank = torch.cuda.current_device()
        _, y_true, _, _ = get_target_samples(args)

        if args.start_rand:
            inputs_1 = torch.randn(
                (args.batch_size, 1, args.resolution, args.resolution),
                requires_grad=True,
                device=device,
                dtype=torch.float,
            )
        else:
            _img = load_prior(args)
            # make init batch
            images = torch.empty(
                size=(
                    args.batch_size,
                    args.channel,
                    args.resolution,
                    args.resolution,
                )
            )
            for i in range(args.batch_size):
                images[i] = _img
            inputs_1 = images.to(device)
            inputs_1.requires_grad_(True)

        if args.init_target_rand:
            targets_in = torch.rand(
                (args.batch_size, 2),
                requires_grad=True,
                device=device,
                dtype=torch.float,
            )   
        else:
            targets_in = torch.tensor(y_true, requires_grad=True, dtype=torch.float, device=device)

        iteration = -1
        iterations_per_layer = args.iterations

        optimizer = torch.optim.Adam(                                                     
            [inputs_1, targets_in],                                                       
            lr=args.lr,
            betas=[0.9, 0.9],
            eps=1e-8,
        )
        lr_scheduler = self.lr_cosine_policy(args.lr, 100, iterations_per_layer)          

        local_trainer = self.create_trainer(
            args=args,
            network=network,
            inputs=inputs_1,                                     
            targets=targets_in,
            criterion=criterion,
            device=torch.device("cuda"),
        )
        for iteration_loc in tqdm(range(iterations_per_layer)):
            iteration += 1
            lr_scheduler(optimizer, iteration_loc, iteration_loc)
                    
            inputs = inputs_1.cuda()
            optimizer.zero_grad()
            network.zero_grad()
            network.train()
            loss_var_l1, loss_var_l2 = self.img_prior(inputs)
            loss_l2 = torch.norm(                                                          
                inputs.view(args.batch_size, -1), dim=1
            ).mean()
            loss_aux = (
                args.tv_l2 * loss_var_l2
                + args.tv_l1 * loss_var_l1
                + args.l2 * loss_l2
            )
            loss = loss_aux
            if args.grad_l2 > 0:
                new_grad = self.sim_local_updates(                                            
                    args=args,
                    trainer=local_trainer,
                    network=network,
                    inputs=inputs,
                    targets=targets_in,
                    use_sigmoid=True,                                                          
                    use_softmax=False,
                )
                loss_grad = 0
                for a, b in zip(new_grad, self.grad_lst):
                    loss_grad += args.grad_l2 * (torch.norm(a - b[1]))
                    # cos_loss = 1 - torch.cosine_similarity(a.view(-1), b[1].view(-1), dim=0).sum()
                    # loss_grad += args.grad_cos * cos_loss
                loss = loss + loss_grad

            # add batch norm loss                                                       
            bn_hooks = []
            self.model_bn.train()
            for name, module in self.model_bn.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    bn_hooks.append(
                        DeepInversionFeatureHook(
                            module=module,
                            bn_stats=self.bn_stats,
                            name=name,
                        )
                    )
            # run forward path once to compute bn_hooks
            self.model_bn(inputs)
            loss_bn_tmp = 0
            for hook in bn_hooks:
                loss_bn_tmp += hook.r_feature
                hook.close()
            loss_bn = args.original_bn_l2 * loss_bn_tmp
            loss += loss_bn
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if local_rank == 0:
                if iteration % save_every == 0:
                    print(f"------------iteration {iteration}----------")
                    print(f"total loss {loss.item()}")
                    print(
                        f"mean targets {torch.mean(targets_in, 0).detach().cpu().numpy()}"
                    )
                    print(f"gradient loss {loss_grad.item()}")
                    try:
                        print(f"bn matching loss {loss_bn.item()}")
                    except:
                        print(f"bn matching loss {loss_bn}")
                    print(
                        f"tvl2 loss {args.tv_l2 * loss_var_l2.item()}"
                    )
            best_inputs = inputs.clone()
            if iteration % save_every == 0 and (save_every > 0):

                self.save_results(                                                    
                    images=best_inputs, targets=targets_in, name="recon"
                )
                # save reconstruction collage
                torchvision.utils.save_image(
                    best_inputs,
                    os.path.join(self.save_path, "recon.png"),
                    normalize=True,
                    scale_each=True,
                    nrow=int(int(args.batch_size) ** 0.5),
                )
            if args.energy_l2 > 0.0:
                inputs_noise_add = torch.randn(inputs.size(), device=device)
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    break
                std = args.energy_l2 * current_lr
                if iteration % save_every == 0:
                    if local_rank == 0:
                        print(
                            f"Energy method waken up, "
                            f"adding Gaussian of std {std}"
                        )
                inputs.data = inputs.data + inputs_noise_add * std

        if save_every > 0:
            self.save_results(images=best_inputs, targets=targets_in, name="recon")

        optimizer.state = collections.defaultdict(dict)

        return best_inputs, targets_in

    def sim_local_updates(
        self,
        args,
        trainer,
        network,
        inputs,
        targets,
        use_softmax=False,
        use_sigmoid=True,
    ):

        params_before = deepcopy(network.state_dict())                 
        trainer.network.load_state_dict(params_before)
        
        if use_softmax:
            targets = torch.softmax(targets, dim=-1)
        if use_sigmoid:
            targets = torch.sigmoid(targets)

        trainer.inputs = inputs
        trainer.targets = targets

        if args.local_optim == "sgd":
            optimizer = torch.optim.SGD(network.parameters(), args.lr_local)
        elif args.local_optim == "adam":
            optimizer = torch.optim.Adam(network.parameters(), args.lr_local)

        trainer.optimizer.load_state_dict(optimizer.state_dict())
        trainer.optimizer.zero_grad()
        trainer.network.zero_grad()
        trainer.run()
        params_after = trainer.network.state_dict()
        new_grad = []
        for name, _ in network.named_parameters():
            new_grad.append(params_after[name] - params_before[name])
        return new_grad

    def create_trainer(self, args, network, inputs, targets, criterion, device=None):
        if device is None:
            device = torch.device("cuda")

        if args.local_optim == "sgd":
            optimizer = torch.optim.SGD(network.parameters(), args.lr_local)
        elif args.local_optim == "adam":
            optimizer = torch.optim.Adam(network.parameters(), args.lr_local)

        optimizer.zero_grad()
        trainer = InversionSupervisedTrainer(
            device=device,
            max_epochs=args.local_epoch,
            inputs=inputs,
            targets=targets,
            network=network,
            optimizer=optimizer,
            loss_function=criterion,
        )
        return trainer

    def img_prior(self, inputs_jit):                                                              
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var_l2 = (
            torch.norm(diff1)
            + torch.norm(diff2)
            + torch.norm(diff3)
            + torch.norm(diff4)
        )
        loss_var_l1 = (
            (diff1.abs() / 255.0).mean()
            + (diff2.abs() / 255.0).mean()
            + (diff3.abs() / 255.0).mean()
            + (diff4.abs() / 255.0).mean()
        )
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def create_folder(self, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)

    def lr_policy(self, lr_fn):
        def _alr(optimizer, iteration, epoch):
            lr = lr_fn(iteration, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return _alr

    def lr_cosine_policy(self, base_lr, warmup_length, epochs):
        def _lr_fn(iteration, epoch):
            if epoch < warmup_length:
                lr = base_lr * (epoch + 1) / warmup_length
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            return lr

        return self.lr_policy(_lr_fn)

    def save_results(self, images, targets, name="recon"):

        # save reconstructed images
        for id in range(images.shape[0]):
            img = images[id, ...]

            save_name = f"{name}_{id}.png"
            place_to_store = os.path.join(self.save_path, save_name)

            image_np = img.data.cpu().numpy()
            image_np = image_np.transpose((1, 2, 0))
            image_np = np.array(
                (image_np - np.min(image_np))
                / (np.max(image_np) - np.min(image_np))
            )
            plt.imsave(place_to_store, image_np)

        # save reconstructed targets
        place_to_store = os.path.join(self.save_path, f"{name}_targets.json")

        with open(place_to_store, "w") as f:
            json.dump(targets.detach().cpu().numpy().tolist(), f, indent=4)


class InversionSupervisedTrainer():

    def __init__(
        self,
        device,
        max_epochs,
        inputs,
        targets,
        network,
        optimizer,
        loss_function,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.inputs = inputs
        self.targets = targets
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function

    def run(self,):

        inputs, targets = self.inputs, self.targets

        self.network.train()                                                                   
        self.network.zero_grad()
        self.optimizer.zero_grad()

        out = self.network(inputs)
        loss = self.loss_function(out, targets)
        loss.backward(retain_graph=True)
        self.optimizer.step()

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
        if self.bn_stats is None:
            var_feature = torch.norm(module.running_var.data - var, 2)
            mean_feature = torch.norm(module.running_mean.data - mean, 2)
        else:
            var_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_var"], device=input[0].device
                )
                - var,
                2,
            )
            mean_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_mean"], device=input[0].device
                )
                - mean,
                2,
            )

        rescale = 1.0
        self.r_feature = mean_feature + rescale * var_feature                               
        self.mean = mean
        self.var = var

    def close(self):
        self.hook.remove()
