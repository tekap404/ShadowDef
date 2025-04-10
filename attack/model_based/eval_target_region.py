import os, sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import argparse
from metric import save_results_with_metric
from util import set_seed, str2bool
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad_inversion')
    parser.add_argument('--cal_target', type=str2bool, default='False')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    available_config = ['chestXray_config', 'eye_config', 'prostate_config', 'face_config']
    parser.add_argument("-c", "--config", choices=available_config, default="chestXray_config", help="config filename")
    parser.add_argument('--root', default='./', type=str)
    parser.add_argument('--output_path', default='../output_attack/client8_prec100', type=str)
    parser.add_argument('--global_model_path', default=None, type=str)

    parser.add_argument('--pretrained_size', nargs='+',                         
                        type=int, default=[1, 3, 224, 224])
    parser.add_argument('--client_id', type=int, default=8)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--perc_cam', type=float, default=0.3)



    args = parser.parse_args()
    args.config = args.config
    args.cal_target = args.cal_target
    args.batch_size = args.pretrained_size[0]
    args.channel = args.pretrained_size[1]
    args.resolution = args.pretrained_size[2]
    

    set_seed(args.seed)

    if args.config == 'chestXray_config':
        from dataset_chestXray import load_rec_imgs, load_nonorm_prior, load_dataloader, load_all_dataloader
        args.template_path = '../data/ChestX-ray14/mean_img.png'
    elif args.config == 'eye_config':
        from dataset_eye import load_rec_imgs, load_nonorm_prior, load_dataloader, load_all_dataloader
        args.template_path = '../data/EyePACS_AIROGS/mean_img.png'

    rec_x = load_rec_imgs(args)
    prior = load_nonorm_prior(args)
    data_loader = load_dataloader(args)
    all_data_loader = load_all_dataloader(args)
    save_results_with_metric(args, rec_x, prior, data_loader, all_data_loader)