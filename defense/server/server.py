from monai.networks.nets.torchvision_fc import TorchVisionFCModel
from utils.dataset_chestXray import get_chestXray_dataloader
from utils.dataset_eye import get_eye_dataloader
import torch
import os
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy
import torchmetrics

def init_all_clients(clients):
    with torch.no_grad():
        global_model = copy.deepcopy(clients[0].model.state_dict())

        avg_dict = {}
        for k, v in global_model.items():
            avg_dict[k] = v.cpu()

    for client in clients:
        client.load_weights(avg_dict)

def weighted_model_avg(clients, weight, server_model, update_client=True):
    with torch.no_grad():
        print('averaging models... weight: %s' % str(weight))
        avg_dict = {}

        for k, v in clients[0].model.state_dict().items():
            if 'running_mean' in k or 'running_var' in k:
                avg_dict[k] = torch.zeros(clients[0].running_mean_var.state_dict()[k].size(), 
                    dtype=torch.float32, layout=clients[0].running_mean_var.state_dict()[k].layout)
            else:
                avg_dict[k] = torch.zeros(v.size(), dtype=torch.float32, layout=v.layout)

        for cid in range(len(clients)):
            for k, v in clients[cid].model.state_dict().items():
                if 'num_batches_tracked' not in k:
                    if 'running_mean' in k or 'running_var' in k:
                        avg_dict[k] += clients[cid].running_mean_var.state_dict()[k].cpu() * weight[cid]
                    else:
                        avg_dict[k] += v.cpu() * weight[cid]

    if update_client:
        for client in clients:
            client.load_weights(avg_dict)
    
    return avg_dict

def weighted_model_avg_with_server_update(grad_list, server, clients, weight, r, given_lr=None, update_client=True):
    
    with torch.no_grad():
        model_delta_dict = {}

        # init delta dict
        for k, v in clients[0].model.state_dict().items():
            model_delta_dict[k] = torch.zeros(v.size(), dtype=torch.float32, layout=v.layout)

        # collect weighted model delta from each clients
        for cid in range(len(clients)):
            for k, v in clients[cid].model.state_dict().items():
                if 'num_batches_tracked' not in k:
                    model_delta_dict[k] += weight[cid] * (grad_list[cid][k].cpu())

    updated_server_dict = server.server_update(model_delta_dict, r, given_lr)
    with torch.no_grad():
        avg_dict = {}
        for k, v in updated_server_dict.items():
            avg_dict[k] = v.cpu()

        for k, v in clients[0].model.state_dict().items():
            if 'running_mean' in k or 'running_var' in k:
                avg_dict[k] = torch.zeros(clients[0].running_mean_var.state_dict()[k].size(), 
                    dtype=torch.float32, layout=clients[0].running_mean_var.state_dict()[k].layout)

        for cid in range(len(clients)):
            for k, v in clients[cid].model.state_dict().items():
                if 'running_mean' in k or 'running_var' in k:
                    avg_dict[k] += clients[cid].running_mean_var.state_dict()[k].cpu() * weight[cid]

    if update_client:
        for client in clients:
            client.load_weights(avg_dict)

    return avg_dict

class MyServer:
    def __init__(self, cfg, df, client_id=0, server_gpu=0, net_dataidx_map=None, serial=True):
        self.cfg = cfg
        self.df = df
        self.client_id = client_id
        self.gpu_id = server_gpu

        if serial:
            self.gpu_id = 0
        self.weights = None

        self.init_model()
        self.init_loaders(net_dataidx_map)   # get val & test sets from all clients
        self.init_optimizer()

    def init_model(self):
        self.model = TorchVisionFCModel(model_name='resnet18', num_classes=2, bias=True, pretrained=True).cuda(self.gpu_id)
        
        output_path = os.path.join(self.cfg.output_dir, self.cfg.exp_name)
        os.makedirs(os.path.join(output_path, 'model/'), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_path, f'model/global_model_init.pth'))

    def init_loaders(self, net_dataidx_map):
        cfg = self.cfg
        
        if self.cfg.config == 'chestXray_config':
            _, self.val_loader, self.test_loader= get_chestXray_dataloader(cfg, self.df, net_dataidx_map[0], 0)
        elif self.cfg.config == 'eye_config':
            _, self.val_loader, self.test_loader= get_eye_dataloader(cfg, self.df, net_dataidx_map[0], 0)

    def init_optimizer(self):
        if self.cfg.optim_server == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr_server, momentum=self.cfg.momentum_server)
        elif self.cfg.optim_server == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.cfg.lr_server, eps=0.001)
        elif self.cfg.optim_server == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0, 0), lr=self.cfg.lr_server, eps=0.001)

    def load_weights(self, state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.cuda(self.gpu_id)
        self.model.load_state_dict(state_dict)

    def adjust_lr(self, r, given_lr=None):
        if self.cfg.lr_schedule_server == 'constant':
            cur_lr = self.cfg.lr_server
        elif self.cfg.lr_schedule_server == 'cosine':
            lr_max = self.cfg.lr_server
            lr_min = 0
            cur_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(r / self.cfg.epochs * math.pi))
        elif self.cfg.lr_schedule_server == 'step':
            cur_lr = self.cfg.lr_server
            for s in self.cfg.lr_step_server:
                if r >= s:
                    cur_lr = cur_lr * 0.1
        if given_lr is not None: # mannual adjust by given lr
            cur_lr = given_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

    # use global optimizer & global gradient to update global model
    def server_update(self, model_delta_dict, cur_round, given_lr=None):
        self.model.train()
        print("Starting server update round %d...\n" % cur_round)

        self.adjust_lr(cur_round, given_lr)
        self.optimizer.zero_grad()

        # Apply the update to the model. We must multiply weights_delta by -1.0 to
        # view it as a gradient that should be applied to the server_optimizer.
        for name, param in self.model.named_parameters():
            param.grad = -1.0*model_delta_dict[name].cuda(self.gpu_id)
        self.optimizer.step()

        print("Round %d, lr: %.5f\n" % (cur_round, self.optimizer.param_groups[-1]['lr']))

        return self.model.state_dict()

    def save_model_init(self, cur_round):
        cfg = self.cfg
        output_path = os.path.join(cfg.output_dir, cfg.exp_name)
        os.makedirs(os.path.join(output_path, 'model/'), exist_ok=True)
        for e in cfg.save_grad:
            if e == cur_round:
                torch.save(self.model.state_dict(), os.path.join(output_path, f'model/global_model_epoch{cur_round + 1}.pth'))

    def save_model(self, cur_round, metric, best_metric, best_metric_round):
        cfg = self.cfg
        output_path = os.path.join(cfg.output_dir, cfg.exp_name)
        os.makedirs(output_path, exist_ok=True)
        writer = open(os.path.join(output_path, 'server_val.log'), 'a')
        tbwriter = SummaryWriter(os.path.join(output_path, 'runs_server_val/'))

        os.makedirs(os.path.join(output_path, 'model/'), exist_ok=True)
        if metric > best_metric:
            best_metric = metric
            best_metric_round = cur_round + 1
            torch.save(self.model.state_dict(), os.path.join(output_path, 'model/best_global_model.pth'))
            writer.write("saved new best metric model\n")
            print("saved new best metric model\n")
        writer.write(
            "Val: current round: {} current mean metric: {:.4f} best mean metric: {:.4f} at round {}\n".format(
                cur_round + 1, metric, best_metric, best_metric_round
            )
        )
        print(
            "Val: current round: {} current mean metric: {:.4f} best mean metric: {:.4f} at round {}\n".format(
                cur_round + 1, metric, best_metric, best_metric_round
            )
        )
        
        tbwriter.add_scalar('val', metric, cur_round + 1)

    def test_server(self, cur_round):
        cfg = self.cfg
        self.model.eval()
        output_path = os.path.join(self.cfg.output_dir, cfg.exp_name)
        os.makedirs(output_path, exist_ok=True)
        writer = open(os.path.join(output_path, 'server_test.log'), 'a')
        tbwriter = SummaryWriter(os.path.join(output_path, 'runs_server_test/'))
        # metric function
        metric_function = torchmetrics.F1Score(average='micro', task="multiclass", num_classes=2, top_k=1)
        metric_function_class = torchmetrics.F1Score(average='none', task="multiclass", num_classes=2, top_k=1)

        with torch.no_grad():
            for test_data in tqdm(self.test_loader):
                test_images, test_labels = test_data[0].cuda(self.gpu_id), test_data[1].cuda(self.gpu_id)
                outputs = self.model(test_images)
                
                _, pred_label = torch.max(outputs.data, 1)
                metric_function.update(pred_label.cpu(), test_labels.cpu())
                metric_function_class.update(pred_label.cpu(), test_labels.cpu())
            metric = metric_function.compute().item()
            metric_class = metric_function_class.compute()
            
            writer.write(
                "Test: current round: {} current mean metric: {:.4f}\n".format(
                    cur_round + 1, metric,
                )
            )
            print(
                "Test: current round: {} current mean metric: {:.4f}\n".format(
                    cur_round + 1, metric,
                )
            )
            for c in range(len(metric_class)):
                writer.write(
                    "Test: current round: {} class: {} metric: {:.4f}\n".format(
                        cur_round + 1, c, metric_class[c],
                    )
                )
                print(
                    "Test: current round: {} class: {} metric: {:.4f}\n".format(
                        cur_round + 1, c, metric_class[c],
                    )
                )
            tbwriter.add_scalar('test', metric, cur_round + 1)
            for c in range(len(metric_class)):
                tbwriter.add_scalar(f'test_class_{c}', metric_class[c], cur_round + 1)
        tbwriter.close()

        return metric

