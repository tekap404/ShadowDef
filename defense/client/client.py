from monai.networks.nets.torchvision_fc import TorchVisionFCModel
from utils.dataset_chestXray import get_chestXray_dataloader
from utils.dataset_eye import get_eye_dataloader
import torch
import os
import torch.nn as nn
import math
from tqdm import tqdm
import copy
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from net.GAN.load_pretrained_GAN import load_pretrained_GAN
import numpy as np
from utils.utils import renorm_resize, DeepInversionFeatureHook
from utils.loss import ShadowLoss, Rec_Loss, Rec_Loss_nomean
from net.GAN.networks_stylegan3 import Generator
from utils.gradcam import GradCAMpp, visualize_cam, get_binary_map
from utils.his_equ import histogram_equalization
import gc

class MyClient:
    def __init__(self, cfg, df, client_id, net_dataidx_map, serial=False):
        self.cfg = cfg
        self.df = df
        self.client_id = client_id
        self.gpu_id = client_id % cfg.gpu
        if serial:
            self.gpu_id = 0

        self.init_model()
        # divide train & val dataloader for each client according to pre-defined pairs (client id, data)
        self.init_loaders(client_id, net_dataidx_map)      
        self.init_optimizer()
        self.init_shadow_scheduler()
        self.init_z()
        self.init_noise_mask()
        
        self.running_mean_var = copy.copy(self.model)
        self.change_bn_flag = 0

    def init_model(self):
        self.model = TorchVisionFCModel(model_name='resnet18', num_classes=2, bias=True, pretrained=True).cuda(self.gpu_id)
        self.shadow_model_tmp = load_pretrained_GAN(self.cfg.GAN_path)

        self.shadow_model = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=256,
            img_channels=3,
        ).cuda(self.gpu_id)
            
    def init_loaders(self, net_id, net_dataidx_map):
        cfg = self.cfg
        dataidxs = net_dataidx_map[net_id]
        
        self.ds_len = len(dataidxs)
        if self.cfg.config == 'chestXray_config':
            self.train_loader, self.val_loader, _ = get_chestXray_dataloader(cfg, self.df, dataidxs, net_id)
        elif self.cfg.config == 'eye_config':
            self.train_loader, self.val_loader, _ = get_eye_dataloader(cfg, self.df, dataidxs, net_id)
        print('client {}: train using {} data points'.format(net_id, len(self.train_loader.dataset)))

    def init_optimizer(self):
        if self.cfg.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.client[self.client_id]['lr'], momentum=self.cfg.momentum)#, weight_decay = 0.0, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.client[self.client_id]['lr'], weight_decay = self.cfg.weight_decay)
        self.shadow_optimizer = torch.optim.Adam(self.shadow_model.parameters(), lr=self.cfg.client[self.client_id]['GAN_lr'], weight_decay = self.cfg.weight_decay)

    def init_shadow_scheduler(self):
        G_total_iter = self.cfg.epochs
        self.G_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.shadow_optimizer, 
            milestones=[G_total_iter // i for i in range(self.cfg.epochs, 1)], 
            gamma=0.8)
        
    def init_z(self):
        def get_name(name_all, loader):
            iter_tmp = iter(loader)
            for _ in range(len(loader)):
                _, _, name = next(iter_tmp)
                name_all.extend(name)
            return name_all

        if self.cfg.z_path is None or self.cfg.no_pretrained_z:
            name_all = []
            name_all = get_name(name_all, self.train_loader)
            name_all = get_name(name_all, self.val_loader)
            self.z_dict = {name: torch.randn([1, self.shadow_model.z_dim]) for name in name_all}
        else:
            self.z_dict = torch.load(self.cfg.z_path + f'client_{self.client_id}_z.pt')

    def init_latent_z(self):
        def get_name(name_all, loader):
            iter_tmp = iter(loader)
            for _ in range(len(loader)):
                _, _, name = next(iter_tmp)
                name_all.extend(name)
            return name_all

        name_all = []
        name_all = get_name(name_all, self.train_loader)

        self.latent_z_dict = {}
        for name in name_all:
            for i in range(self.cfg.end_layer):
                self.latent_z_dict[name + '_layer' + str(i)] = None

    def init_noise_mask(self):
        def get_name(name_all, loader):
            iter_tmp = iter(loader)
            for _ in range(len(loader)):
                _, _, name = next(iter_tmp)
                name_all.extend(name)
            return name_all

        name_all = []
        name_all = get_name(name_all, self.train_loader)
        self.noise_mask_dict = {name: torch.randn([3, self.cfg.img_size, self.cfg.img_size]).cuda(self.gpu_id) for name in name_all}

    def load_weights(self, state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
    
    def adjust_lr(self, r, given_lr=None):
        if self.cfg.lr_schedule == 'constant':
            cur_lr = self.cfg.client[self.client_id]['lr']
        elif self.cfg.lr_schedule == 'cosine':
            lr_max = self.cfg.client[self.client_id]['lr']
            lr_min = 0
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(r / self.cfg.local_round * math.pi))
        elif self.cfg.lr_schedule == 'step':
            cur_lr = self.cfg.client[self.client_id]['lr']
            for s in self.cfg.lr_step:
                if r >= s:
                    cur_lr = cur_lr * 0.1
        if given_lr is not None: # mannual adjust by given lr
            cur_lr = given_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

    def update_task_model(self, cfg, inputs, labels, loss_function, progress_bar, cur_round, iters):

        outputs = self.model(inputs)
        loss = loss_function(outputs, labels)
        exact_grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        target_gradient = [grad.detach() for grad in exact_grad]
        layer_name = [k for k, _ in self.model.named_parameters()]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        progress_bar.set_description(f"client: {self.client_id} iteration {iters + cur_round * cfg.local_round * self.ds_len / cfg.batch_size} loss: {loss.item():.4f} lr: {self.optimizer.param_groups[0]['lr']:.6f}")
        iters += 1

        return iters, target_gradient, layer_name

    # update shadow model
    def train_shadow_model_gias(self, cur_round, bs_id, inputs, labels, names, grad_list, bn_dict, global_model, loss_function, optimizer):

        loss_func_shadow = ShadowLoss(self.cfg)
        
        z = torch.randn([inputs.shape[0], self.shadow_model.z_dim], device=inputs.device)
        for i, n in enumerate(names):
            z[i] = self.z_dict[n][0].to(inputs.device)
        z.requires_grad_(False)

        G_iter = self.cfg.client[self.client_id]['GAN_round']
        
        self.shadow_model.train()
        self.shadow_model.mapping.requires_grad_(False)
        self.shadow_model.synthesis.requires_grad_(True)

        # whole network
        # optimize G
        mse_list = []
        progress_bar = tqdm(range(G_iter))
        for shadow_iter, _ in enumerate(progress_bar):
            # reset
            model_tmp = copy.deepcopy(global_model)
            model_tmp.eval()
            # optim
            if self.cfg.optimizer == 'sgd':
                optimizer_tmp = torch.optim.SGD(model_tmp.parameters(), lr=self.cfg.client[self.client_id]['lr'], momentum=self.cfg.momentum)
            else:
                optimizer_tmp = torch.optim.Adam(model_tmp.parameters(), lr=self.cfg.client[self.client_id]['lr'], weight_decay = self.cfg.weight_decay)
            optimizer_tmp.load_state_dict(optimizer.state_dict())
            optimizer_tmp.zero_grad()
            self.shadow_optimizer.zero_grad()

            # BN
            bn_hooks = []
            for name, module in model_tmp.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    bn_hooks.append(
                        DeepInversionFeatureHook(
                            module=module,
                            bn_stats=bn_dict,
                            name=name,
                        )
                    )

            # random input for shadow model
            c = [np.random.randint(100000) for _ in range(self.cfg.client[self.client_id]['batch_size'])]
            c = torch.from_numpy(np.stack(c)).to(inputs.device)

            # forward GAN
            pseudo_img = self.shadow_model(z, c)
            pseudo_img = renorm_resize(self.cfg, pseudo_img, [-1, 1])

            # forward task model
            outputs = model_tmp(pseudo_img)
            loss = loss_function(outputs, labels)

            # record model update
            gradient = torch.autograd.grad(loss, model_tmp.parameters(), retain_graph=True, create_graph=True, only_inputs=True)
            shadow_grad = [grad for grad in gradient]

            loss_shadow, loss_grad, loss_bn, loss_var, loss_norm, loss_rec = loss_func_shadow(pseudo_img, grad_list, shadow_grad, bn_hooks, inputs)
            progress_bar.set_description(f"loss(All/Grad/BN/var/norm/rec): {loss_shadow.item():.6f}, {loss_grad:.6f}, {loss_bn:.6f}, {loss_var:.6f}, {loss_norm:.6f}, {loss_rec:.6f}, lr: {self.shadow_optimizer.param_groups[0]['lr']:.6f} \n")
            loss_shadow.backward()
            self.shadow_optimizer.step()
        
    def optimize_z(self, ):

        self.shadow_model.eval()
        self.shadow_model.mapping.requires_grad_(False)
        self.shadow_model.synthesis.requires_grad_(True)
        loss_func = Rec_Loss(self.cfg)

        def opti_func(self, target_iter):

            # for train data
            progress_bar = tqdm(range(len(target_iter)))
            tr_it = iter(target_iter)
            for idx, _ in enumerate(progress_bar):
                batch_data = next(tr_it)
                inputs, name = batch_data[0].cuda(self.gpu_id), batch_data[2]
                
                z = torch.randn([inputs.shape[0], self.shadow_model.z_dim], device=inputs.device)
                z.requires_grad_(True)
                z_optimizer = torch.optim.AdamW([z], lr=self.cfg.z_lr)
                z_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    z_optimizer, 
                    milestones=[self.cfg.z_iter // 2.667, self.cfg.z_iter // 1.6, self.cfg.z_iter // 1.142], gamma=0.5
                )

                mse_list = []
                count = 0
                # optimize z
                for z_iter in tqdm(range(self.cfg.z_iter)):
                    c = [np.random.randint(100000) for _ in range(inputs.shape[0])]
                    c = torch.from_numpy(np.stack(c)).to(inputs.device)

                    # forward GAN
                    pseudo_img = self.shadow_model(z, c)
                    pseudo_img = renorm_resize(self.cfg, pseudo_img, [-1, 1])

                    if z_iter > 50:
                        if mse_list[-1] > mse_list[-2]:
                            count += 1
                            if count >= self.cfg.z_patience:
                                break
                        else:
                            count = 0
                        
                    loss = loss_func(pseudo_img, inputs)
                    loss.backward()
                    z_optimizer.step()
                    z_scheduler.step()

                for i, n in enumerate(name):
                    self.z_dict[n] = z[i].unsqueeze(0).detach().to('cpu')
        
        opti_func(self, self.train_loader)
        opti_func(self, self.val_loader)
        torch.save(self.z_dict, self.cfg.output_dir + self.cfg.exp_name + f'/client_{self.client_id}_z.pt')

    def update_noise(self, cur_round, bs_id, inputs, labels, names, grad_list, layer_name, updated_model, global_model, loss_function, optimizer):
        loss_func = Rec_Loss_nomean(self.cfg)
        
        z = torch.randn([inputs.shape[0], updated_model.z_dim], device=inputs.device)
        for i, n in enumerate(names):
            z[i] = self.z_dict[n][0].to(inputs.device)
        z.requires_grad_(False)

        updated_model.eval()

        # reset
        model_tmp = copy.deepcopy(global_model)
        model_tmp.eval()

        # random input for shadow model
        c = [np.random.randint(100000) for _ in range(self.cfg.client[self.client_id]['batch_size'])]
        c = torch.from_numpy(np.stack(c)).to(inputs.device)

        # forward GAN
        pseudo_img = updated_model(z, c)
        pseudo_img = renorm_resize(self.cfg, pseudo_img, [-1, 1])
        
        # rec loss
        rec_loss = loss_func(pseudo_img, inputs)
        with torch.no_grad():
            rec_loss = torch.softmax(rec_loss.view(rec_loss.shape[0], -1)/20, dim=1).view(rec_loss.shape)
            # get image noise 
            img_noise1 = 1/rec_loss
            if self.cfg.noise_equ:
                # perform histogram equalization
                img_noise2 = histogram_equalization(img_noise1)
            img_noise2 = torch.softmax( img_noise2.view(rec_loss.shape[0], -1)/20, dim=1).view(rec_loss.shape)

        return img_noise2

    def add_noise(self, cfg, inputs, img_noise, mask_pp, cur_round, bs_id, sample_id):

        # post-process gradcam (pick out salience map & sharpen)
        mask_pp = mask_pp.squeeze(1)
        mask_tmp = mask_pp.view(-1)
        _, indices = mask_tmp.topk(int(mask_tmp.numel() * cfg.perc_cam))
        binary_map = torch.zeros_like(mask_tmp)
        binary_map.scatter_(0, indices, 1)
        binary_map = binary_map.view(mask_pp.shape)
        post_gradcam = binary_map * mask_pp
        post_gradcam = torch.softmax(post_gradcam.view(-1)/20, dim=0).view(post_gradcam.shape)

        # calculate final_noise
        alpha = min(cfg.max_cam_alpha, max(cfg.min_cam_alpha, (cur_round/cfg.epochs)))
        sign_map = torch.sign(img_noise)
        if cfg.sub_cam:
            noise_4 = img_noise - alpha * sign_map *post_gradcam    # change to zero  
        else:
            noise_4 = img_noise

        if cfg.noise_increase:
            reweight = cfg.noise_rescale * math.exp(cur_round/cfg.epochs)   # increase noise amplitude since GIA usually becomes weaker in late epochs
            noise = abs(inputs.max()/noise_4.max()*reweight) * noise_4
        else:
            noise = abs(inputs.max()/noise_4.max()*cfg.noise_rescale) * noise_4

        # add noise & post-processing
        noisy_inputs = inputs + noise
        return noisy_inputs

    def true_update_shadow_model(self, shadow_model, updated_model, cur_round, bs_id, inputs, labels, names, grad_list, bn_dict, global_model, loss_function, optimizer):
        # EMA
        for ((k1, _), (_, v2), (_, v3)) in zip(self.shadow_model.named_parameters(), updated_model.named_parameters(), shadow_model.named_parameters()):
            self.shadow_model.state_dict()[k1].copy_(self.cfg.ema_shadow *v2.data + (1-self.cfg.ema_shadow) *v3.data)
        
    # local training
    def train_round(self, cur_round, given_lr, global_model, grad_list):
        
        self.model.train()
        cfg = self.cfg
        iters, epochs = 0, 0
        loss_function = nn.CrossEntropyLoss()
        
        os.makedirs(os.path.join(self.cfg.output_dir, self.cfg.exp_name), exist_ok=True)
        writer = open(os.path.join(self.cfg.output_dir, self.cfg.exp_name, 'client_%d.log' % self.client_id), 'a')
        writer.write("Starting round %d...\n" % cur_round)
        print("Starting round %d...\n" % cur_round)

        self.adjust_lr(cur_round, given_lr)
        writer.write('Train {} local epoch for client {}'.format(cfg.local_round, self.client_id))
        print('Train {} local epoch for client {}'.format(cfg.local_round, self.client_id))

        # grad_cam
        model_cam_dict = dict(arch=global_model, layer_name='features.7.1', input_size=(224, 224))
        gradcampp = GradCAMpp(model_cam_dict, verbose=False)

        while epochs < cfg.local_round:
            print(f'Client {self.client_id} Current local epoch {epochs}')
            progress_bar = tqdm(range(len(self.train_loader)))
            tr_it = iter(self.train_loader)
            for bs_id, _ in enumerate(progress_bar):
                batch_data = next(tr_it)
                inputs, labels, names = batch_data[0].cuda(self.gpu_id), batch_data[1].cuda(self.gpu_id), batch_data[2]

                if cfg.pseudo_train_img:
                    # pseudo train
                    model_tmp = copy.deepcopy(self.model)
                    optimizer_tmp = copy.deepcopy(self.optimizer)
                
                iters_tmp, target_gradient, layer_name = self.update_task_model(cfg, inputs, labels, loss_function, progress_bar, cur_round, iters)
                if not cfg.pseudo_train_img:
                    iters = iters_tmp

                # cal model update
                grad_dict = {}
                for k, v in self.model.state_dict().items():
                    grad_dict[k] = v.cpu() - global_model.state_dict()[k].cpu()

                # cal BN statistics
                bn_stats = {}
                for param_name in grad_dict.keys():
                    if "bn" in param_name or "batch" in param_name or "running" in param_name:
                        bn_stats[param_name] = global_model.state_dict()[param_name].cpu() + grad_dict[param_name]
                for n in bn_stats.keys():
                    if "running" in n:
                        xt = (bn_stats[n] - (1 - 0.1) * global_model.state_dict()[n].cpu()) / 0.1     # xt = (updated running mean/var - 0.9*history) / 0.1=latest
                        bn_stats[n] = xt.to(inputs.device)
                
                if cur_round < cfg.shadow_final_epoch:
                    # pseudo update shadow model
                    print('Pseudo update shadow model .......')
                    shadow_model = copy.deepcopy(self.shadow_model)
                    shadow_optimizer_tmp = copy.deepcopy(self.shadow_optimizer)
                    if self.cfg.shadow_train:
                        self.train_shadow_model_gias(cur_round, bs_id, inputs, labels, names, target_gradient, bn_stats, global_model, loss_function, copy.deepcopy(self.optimizer))
                    updated_model = copy.deepcopy(self.shadow_model)
                    self.shadow_optimizer.load_state_dict(shadow_optimizer_tmp.state_dict())
                            
                    # get noise
                    # grad_noise
                    print('Get noise .......')
                    img_noise = self.update_noise(cur_round, bs_id, inputs, labels, names, target_gradient, layer_name, updated_model, global_model, loss_function, copy.deepcopy(self.optimizer))

                    # update momentum noise
                    for per_img_noise, name in zip(img_noise, names):
                        if cur_round == 0:
                            self.noise_mask_dict[name] = per_img_noise
                        else:
                            self.noise_mask_dict[name] = cfg.noise_momentum * self.noise_mask_dict[name] + (1 - cfg.noise_momentum) * per_img_noise
                else:
                    # load momentum noise
                    img_noise = []
                    for name in names:
                        img_noise.append(self.noise_mask_dict[name])
                    img_noise = torch.stack(img_noise, dim=0).cuda(self.gpu_id)

                # get gradcam of task model
                print('Get CAM .......')
                gradcam_list = []
                for b in range(inputs.shape[0]):
                    mask_pp, _ = gradcampp(inputs[b].unsqueeze(0), class_idx=1) # ∈[0,1], sum!=1

                    ### vis
                    binary_map = get_binary_map(mask_pp, perc=cfg.perc_cam)
                    heatmap_pp, result_pp = visualize_cam(mask_pp, inputs[b])
                    gradcam_list.append(torch.stack([inputs[b].cpu(), heatmap_pp, binary_map, result_pp], 0))
                
                    # post-precess & add noise
                    inputs[b] = self.add_noise(self.cfg, inputs[b], img_noise[b], mask_pp, cur_round, bs_id, b)
                
                if cur_round < cfg.shadow_final_epoch and self.cfg.shadow_train:
                    # update shadow model
                    print('True update shadow model .......')
                    self.true_update_shadow_model(shadow_model, updated_model, cur_round, bs_id, inputs, labels, names, target_gradient, bn_stats, global_model, loss_function, copy.deepcopy(self.optimizer))
                    self.G_scheduler.step()

                if cfg.pseudo_train_img:
                    # true update
                    self.load_weights(model_tmp.state_dict())
                    self.optimizer.load_state_dict(optimizer_tmp.state_dict())
                    iters, target_gradient, layer_name = self.update_task_model(cfg, inputs, labels, loss_function, progress_bar, cur_round, iters)

            epochs += 1
        writer.close()
        
        grad_dict = {}
        for k, v in self.model.state_dict().items():
            grad_dict[k] = v.cpu() - global_model.state_dict()[k].cpu()
            
        grad_list[self.client_id] = grad_dict

        os.makedirs(os.path.join(self.cfg.output_dir, self.cfg.exp_name, f'client{self.client_id}_model'), exist_ok=True)
        for e in cfg.save_grad:
            if e == cur_round:
                torch.save(grad_dict, os.path.join(self.cfg.output_dir, self.cfg.exp_name, f'client{self.client_id}_model', f'client{self.client_id}_grad_epoch{cur_round+1}.pkl'))

        if cur_round < cfg.shadow_final_epoch:
            del batch_data, inputs, labels, optimizer_tmp, target_gradient, bn_stats, shadow_model, \
            shadow_optimizer_tmp, updated_model, img_noise, mask_pp
        else:
            del batch_data, inputs, labels, optimizer_tmp, target_gradient, bn_stats, img_noise, mask_pp
        gc.collect()
        torch.cuda.empty_cache()

        return grad_list

    def validation_round(self, cur_round, result_queue):
        cfg = self.cfg
        self.model.eval()
        output_path = os.path.join(cfg.output_dir, cfg.exp_name)
        os.makedirs(output_path, exist_ok=True)
        writer = open(os.path.join(output_path, f'client{self.client_id}_val.log'), 'a')
        tbwriter = SummaryWriter(os.path.join(output_path, f'runs_client{self.client_id}_val/'))
        # metric function
        metric_function = torchmetrics.F1Score(average='micro', task="multiclass", num_classes=2, top_k=1)
        metric_function_class = torchmetrics.F1Score(average='none', task="multiclass", num_classes=2, top_k=1)

        with torch.no_grad():
            for val_data in tqdm(self.val_loader):
                val_images, val_labels = val_data[0].cuda(self.gpu_id), val_data[1].cuda(self.gpu_id)
                outputs = self.model(val_images)

                _, pred_label = torch.max(outputs.data, 1)
                metric_function.update(pred_label.cpu(), val_labels.cpu())
                metric_function_class.update(pred_label.cpu(), val_labels.cpu())
            metric = metric_function.compute().item()
            metric_class = metric_function_class.compute()

            writer.write(
                "Client: {} Val: current round: {} current mean metric: {:.4f}\n".format(
                    self.client_id, cur_round + 1, metric,
                )
            )
            print(
                "Client: {} Val: current round: {} current mean metric: {:.4f}\n".format(
                    self.client_id, cur_round + 1, metric,
                )
            )
            for c in range(len(metric_class)):
                writer.write(
                    "Client: {} Val: current round: {} class: {} metric: {:.4f}\n".format(
                        self.client_id, cur_round + 1, c, metric_class[c]
                    )
                )
                print(
                    "Client: {} Val: current round: {} class: {} metric: {:.4f}\n".format(
                        self.client_id, cur_round + 1, c, metric_class[c]
                    )
                )
            
            tbwriter.add_scalar('val', metric, cur_round + 1)
            for c in range(len(metric_class)):
                tbwriter.add_scalar(f'val_class_{c}', metric_class[c], cur_round + 1)
        
        writer.close()
        tbwriter.close()

        result_queue[self.client_id] = metric

        return result_queue