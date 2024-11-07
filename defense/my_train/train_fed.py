import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import time
import warnings
warnings.filterwarnings("ignore")
from utils.utils import *
from client.client import MyClient
from server.server import MyServer, init_all_clients, weighted_model_avg_with_server_update

def main(cfg):
    
    print(cfg)

    print('# Number of training Clients:{}'.format(cfg.client_num))

    log = cfg.log
    if log:
        log_path = cfg.output_dir + cfg.exp_name
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, 'FL.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        for k in list(vars(cfg).keys()):
            logfile.write('{}: {}\n'.format(k, vars(cfg)[k]))

    if cfg.config == 'chestXray_config':
        from utils.dataset_chestXray import get_partitioned_data
    elif cfg.config == 'eye_config':
        from utils.dataset_eye import get_partitioned_data
    net_dataidx_map, df = get_partitioned_data(cfg, datadir=cfg.data_dir)

    if cfg.gpu > 1:
        serial = False
    else:
        serial = True

    # initialize cliens and server
    clients, client_dicts = [], []
    for cid in range(cfg.client_num):
        clients.append(MyClient(cfg, df, cid, net_dataidx_map, serial=serial))
        md = {}
        md['best_metric'] = -1
        md['best_metric_round'] = -1
        client_dicts.append(md)

    server_dict = {}
    server_dict['best_metric'] = -1
    server_dict['best_metric_round'] = -1
    server = MyServer(cfg, df, client_id=0, server_gpu=cfg.server_gpu, net_dataidx_map=net_dataidx_map, serial=serial)
    init_all_clients(clients)
        
    weight = [client.ds_len for client in clients]
    weight = [e / sum(weight) for e in weight]
    best_val_metric, best_metric_round = 0., 0

    # optimize z before training
    if cfg.z_path is None:
        for client in clients:
            client.optimize_z()
    else:
        grad_list = [None for _ in range(9)]
        for r in range(cfg.epochs):
            print('Start training round %d...' % r)
            start_time = time.time()

            server.save_model_init(r)

            # train round
            for i, client in enumerate(clients):
                grad_list = client.train_round(r, None, server.model, grad_list)

            # model aggregation
            server.load_weights(weighted_model_avg_with_server_update(grad_list, server, clients, weight, r, cfg.lr_server))

            if (r+1) % cfg.eval_epochs == 0:
                # val
                result_queue = [None for _ in range(9)]
                for i, client in enumerate(clients):
                    result_queue = client.validation_round(r, result_queue)
                
                metric = 0.
                for i in range(len(clients)):
                    metric += result_queue[i]
                metric = metric / len(clients)
                
                server.save_model(r, metric, best_val_metric, best_metric_round)
                if metric > best_val_metric:
                    best_val_metric = metric
                    best_metric_round = r + 1

                # test center model
                server.test_server(r)

            end_time = time.time()
            print('time elapsed: %.4f' % (end_time - start_time))





