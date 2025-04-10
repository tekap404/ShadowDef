
import copy
import os
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt
import torch.distributed as dist
from prettytable import PrettyTable
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from monai.networks.nets.torchvision_fc import TorchVisionFCModel
import pandas as pd
from gradcam import GradCAMpp, get_binary_map

def save_imgs(imgs, pretrained_size, dir_path):

    n_channels = pretrained_size[1]
    # save pseudo-samples
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
        plt.title(str(i), fontsize=30)
        plt.axis('off')
        i += 1
    filename = 'closest_batch.jpg'
    path = os.path.join(dir_path, filename)
    plt.savefig(path)
    plt.close()
    print(f' > save image: {path}')

def save_results_with_metric(args, rec_x, prior, data_loader, all_data_loader):
    best_x = compute_metric(args, rec_x, prior, data_loader, all_data_loader)
    save_imgs(best_x, args.pretrained_size, args.output_path)

def compute_metric(args, x_recon, prior, data_loader, all_data_loader):

    metric_list = ['mse', 'psnr', 'lpips', 'ssim', 'rdlv', '1-iip', '3-iip', '5-iip']
    metric = ImageSimilarity(args, metric_list)

    print('inversion metric')

    metrics_batch, best_match_batch = metric.min_similarity(
        x_recon, prior, data_loader, all_data_loader)
    analysis_table, metrics_batch = show_metric(metrics_batch)
    print(f'Metric: '+str(analysis_table))
    save_metric(args, metrics_batch)

    return best_match_batch

def show_metric(metrics_batch):
    column_name = ['T:IDX', 'mse', 'psnr', 'lpips', 'ssim', 'rdlv', '1-iip', '3-iip', '5-iip']
    metric_name = column_name[1:]

    mean_metric = dict()
    for key in column_name:
        if key == 'T:IDX':
            mean_metric[key] = 'mean'
        else:
            mean_metric[key] = 0

    std_metric = dict()
    for key in column_name:
        if key == 'T:IDX':
            std_metric[key] = 'std'
        else:
            std_metric[key] = []

    for i, metrics in enumerate(metrics_batch):

        metrics_batch[i]['T:IDX'] = int(i)
        for key in metric_name:
            mean_metric[key] += metrics[key]
            std_metric[key].append(metrics[key])

    for key in metric_name:
        mean_metric[key] /= len(metrics_batch)
        std_metric[key] = torch.as_tensor(std_metric[key]).std().item()

    metrics_batch.append(mean_metric)
    metrics_batch.append(std_metric)

    table = PrettyTable(column_name)

    for i in range(len(metrics_batch)):
        values = []
        for key in column_name:
            value = metrics_batch[i][key]
            if not isinstance(value, float):
                values.append(value)
            else:
                values.append(f'{value:.4f}')
        table.add_row(values)

    return table, metrics_batch

def save_metric(args, metrics_batch):

    metric_name = ['mse', 'psnr', 'lpips', 'ssim', 'rdlv', '1-iip', '3-iip', '5-iip']
    metric_list = []
    for i in range(len(metrics_batch)):
        values = []
        for key in metric_name:
            value = metrics_batch[i][key]
            values.append(value)
        metric_list.append(values)
    df = pd.DataFrame(data=metric_list, 
                  index=[i+1 for i in range(len(metrics_batch)-2)] + ['mean', 'std'],
                  columns=metric_name)

    if not args.cal_target:
        df.to_csv(os.path.join(args.output_path, 'metric.csv'))
    else:
        df.to_csv(os.path.join(args.output_path, 'metric_target_region.csv'))

class ImageSimilarity():

    def __init__(self, args,
                 metric_list) -> None:

        self.args = args
        self.metric_list = metric_list
        if 'lpips' in metric_list:
            self.lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    def load_model(self):

        model = TorchVisionFCModel(
            model_name='resnet18',
            num_classes=2,
            pretrained=True,
        )
        ckpt = torch.load(self.args.global_model_path)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        remove_head = torch.nn.Sequential()
        model.fc = remove_head
        model.eval()

        return model.cuda()

    # not used
    def batch_similarity(
            self, target_batch,
            recon_batch, T_to_R_map):

        metrics_batch = []

        for j_R, i_T in enumerate(T_to_R_map):
            # 以重构样本顺序为正序
            target = target_batch[i_T]
            recon = recon_batch[j_R]
            metrics = self.compute_similarity(target, recon)
            metrics_batch.append(metrics)

        return metrics_batch

    def min_similarity(self, recon_batch, prior, data_loader, all_data_loader):
        
        metrics_batch = []
        best_match_batch = [None for _ in range(len(recon_batch))]
        
        # 获取所有FL的训练+验证集样本特征
        model = self.load_model()
        if self.args.cal_target:
            model_cam_dict = dict(arch=model, layer_name='features.7.1', input_size=(224, 224))
            gradcampp = GradCAMpp(model_cam_dict, verbose=False)

        deep_features, clientid_list = [], []
        progress_bar = tqdm(range(len(all_data_loader)))
        tr_it = iter(all_data_loader)

        with torch.no_grad():
            for _ in progress_bar:
                batch_data = next(tr_it)
                inputs, client_idx = batch_data[0].cuda(), batch_data[1]
                feature = model(inputs).cpu().squeeze(0)
                deep_features.append(feature)
                clientid_list.append(client_idx)

        for rec_idx, rec_img in enumerate(tqdm(recon_batch)):
            if 'mse' in self.metric_list:                            # 论文的右侧y轴
                min_mse = float('inf')
            if 'psnr' in self.metric_list:
                max_psnr = 0
            if 'lpips' in self.metric_list:
                min_lpips = float('inf')
            if 'ssim' in self.metric_list:
                max_ssim = 0
            if 'rdlv' in self.metric_list:
                max_rdlv = -float('inf')

            progress_bar = range(len(data_loader))
            tr_it = iter(data_loader)
            for _ in progress_bar:
                batch_data = next(tr_it)
                inputs = batch_data[0].squeeze(0)   # data_loader.batch_size=1 (search for per sample)
                ori_inputs = copy.deepcopy(inputs)
                if self.args.cal_target:
                    mask_pp, _ = gradcampp(inputs.cuda().unsqueeze(0), class_idx=1) # ∈[0,1], sum!=1
                    binary_mask = get_binary_map(mask_pp, perc=self.args.perc_cam).cpu()
                    binary_mask = (1-binary_mask).squeeze(0).repeat(3,1,1)

                if self.args.cal_target:
                    metrics = self.compute_similarity(inputs, rec_img, prior, binary_mask)
                else:
                    metrics = self.compute_similarity(inputs, rec_img, prior)

                if 'mse' in self.metric_list and metrics['mse'] < min_mse:
                    min_mse = metrics['mse']

                if 'psnr' in self.metric_list and metrics['psnr'] > max_psnr:
                    max_psnr = metrics['psnr']

                if 'lpips' in self.metric_list and metrics['lpips'] < min_lpips:
                    min_lpips = metrics['lpips']

                if 'ssim' in self.metric_list and metrics['ssim'] > max_ssim:
                    max_ssim = metrics['ssim']

                if 'rdlv' in self.metric_list and metrics['rdlv'] > max_rdlv:
                    max_rdlv = metrics['rdlv']
                    best_match_batch[rec_idx] = ori_inputs

            if 'mse' in self.metric_list:
                metrics['mse'] = min_mse
            if 'psnr' in self.metric_list:
                metrics['psnr'] = max_psnr
            if 'lpips' in self.metric_list:
                metrics['lpips'] = min_lpips
            if 'ssim' in self.metric_list:
                metrics['ssim'] = max_ssim
            if 'rdlv' in self.metric_list:
                metrics['rdlv'] = max_rdlv

            cos_list = []
            recon_feature = model(rec_img.unsqueeze(0).cuda()).squeeze(0)
            for feature in deep_features:
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(recon_feature.view(-1), feature.cuda().view(-1))
                cos_list.append(cos.cpu())
            cos_list = torch.stack(cos_list, dim=0)
            value_top, indices_top = cos_list.topk(10, dim=0, largest=True, sorted=True)

            metrics['1-iip'] = self.caliip(indices_top, clientid_list, value_top, 1)
            metrics['3-iip'] = self.caliip(indices_top, clientid_list, value_top, 3)
            metrics['5-iip'] = self.caliip(indices_top, clientid_list, value_top, 5)

            metrics_batch.append(metrics)
        
        return metrics_batch, best_match_batch

    def compute_similarity(self, target, recon, prior, binary_mask=None):

        metrics = {}

        if 'mse' in self.metric_list:
            metrics['mse'] = self.mean_squared_error(target, recon, binary_mask)

        if 'psnr' in self.metric_list:
            if 'mse' in self.metric_list:
                metrics['psnr'] = \
                    self.peak_signal_noise_ratio(metrics['mse'])
            else:
                mse = self.mean_squared_error(target, recon, binary_mask)
                metrics['psnr'] = \
                    self.peak_signal_noise_ratio(mse)

        if 'lpips' in self.metric_list:
            metrics['lpips'] = self.perceptual_metric(target, recon, binary_mask)

        if 'ssim' in self.metric_list:
            metrics['ssim'] = self.calc_ssim(target, recon, binary_mask)
        
        if 'rdlv' in self.metric_list:
            metrics['rdlv'] = self.calc_rdlv(target, recon, prior)

        return metrics

    def mean_squared_error(self, target, recon, binary_mask=None):

        if binary_mask == None:
            mse = ((target - recon) ** 2).mean()
        else:
            mse = (((target - recon) ** 2)*binary_mask).mean()

        return mse.item()

    def peak_signal_noise_ratio(self, mse, factor=1.0):

        mse = torch.as_tensor(mse)
        return (10 * torch.log10(factor**2 / mse)).item()

    def perceptual_metric(self, target, recon, binary_mask=None):
        """LPIPS: https://github.com/richzhang/PerceptualSimilarity"""
        if target.size()[-1] < 32:
            # 该测量要求输入像素点大于32
            resize_fun = transforms.Resize(32)
        else:
            resize_fun = transforms.Lambda(lambda x: x)

        norm = transforms.Normalize(
                    std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5)) if self.args.channel == 3 else transforms.Normalize(
                    std=(0.5), mean=(0.5))
        resizer = transforms.Compose([
            transforms.ToPILImage(),
            resize_fun,
            transforms.ToTensor(),
            norm,
        ])

        if binary_mask != None:
            target = target * binary_mask
            recon = recon * binary_mask

        target = resizer(target).cuda()
        recon = resizer(recon).cuda()

        lpips_score = self.lpips_fn(target, recon).cpu()

        return lpips_score.item()

    def calc_ssim(self, target, recon, binary_mask=None):
        '''tructural similarity index'''

        if binary_mask != None:
            target = target * binary_mask
            recon = recon * binary_mask

        if target.shape[0] > 1:
            multichannel = True
        else:
            multichannel = False
        tt = transforms.ToPILImage()
        target_np = np.array(tt(target))
        recon_np = np.array(tt(recon))
        ssim_score = ssim(
            target_np, recon_np,
            data_range=255, multichannel=multichannel, channel_axis=2)

        return ssim_score

    def calc_rdlv(self, target, recon, prior):

        if target.shape[0] > 1:
            multichannel = True
        else:
            multichannel = False
        tt = transforms.ToPILImage()
        target_np = np.array(tt(target))
        recon_np = np.array(tt(recon))
        prior_np = np.array(tt(prior))
        ssim_1 = ssim(
            target_np, recon_np,
            data_range=255, multichannel=multichannel, channel_axis=2)
        ssim_2 = ssim(
            target_np, prior_np,
            data_range=255, multichannel=multichannel, channel_axis=2)
        rdlv_score = (ssim_1 - ssim_2) / ssim_2

        return rdlv_score

    def caliip(self, indices_top5, clientid_list, value_top, k):

        # 避免同分
        num = copy.deepcopy(k)
        last = value_top[k-1]
        for tmp in value_top[k:]:
            if tmp == last:
                num += 1
        
        count = 0
        for i in range(num):
            if self.args.client_id == clientid_list[indices_top5[i]]:
                count += 1
            
        iip = count / self.args.batch_size

        return iip

    


def to_img(x):
    '''
    x.shape = (n_imgs, n_channels, H, W)
    '''
    x = x.detach().clone().cpu()

    imgs = []
    for img in x:
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        imgs.append(img)

    imgs = torch.stack(imgs)

    return imgs