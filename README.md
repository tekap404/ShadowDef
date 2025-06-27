# ShadowDef

## Introduction

ShadowDef is a defense framework against gradient inversion attack under federated learning. 

## Citation

```
@article{JIANG2025103673,
title = {Shadow defense against gradient inversion attack in federated learning},
journal = {Medical Image Analysis},
pages = {103673},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103673},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525002208},
author = {Le Jiang and Liyan Ma and Guang Yang}
}
```

## Abstract

Federated learning (FL) has emerged as a transformative framework for privacy-preserving distributed training, allowing clients to collaboratively train a global model without sharing their local data. This is especially crucial in sensitive fields like healthcare, where protecting patient data is paramount. However, privacy leakage remains a critical challenge, as the communication of model updates can be exploited by potential adversaries. Gradient inversion attacks (GIAs), for instance, allow adversaries to approximate the gradients used for training and reconstruct training images, thus stealing patient privacy. Existing defense mechanisms obscure gradients, yet lack a nuanced understanding of which gradients or types of image information are most vulnerable to such attacks. These indiscriminate calibrated perturbations result in either excessive privacy protection degrading model accuracy, or insufficient one failing to safeguard sensitive information. Therefore, we introduce a framework that addresses these challenges by leveraging a shadow model with interpretability for identifying sensitive areas. This enables a more targeted and sample-specific noise injection. Specially, our defensive strategy achieves discrepancies of 3.73 in PSNR and 0.2 in SSIM compared to the circumstance without defense on the ChestXRay dataset, and 2.78 in PSNR and 0.166 in the EyePACS dataset. Moreover, it minimizes adverse effects on model performance, with less than 1% F1 reduction compared to SOTA methods. Our extensive experiments, conducted across diverse types of medical images, validate the generalization of the proposed framework. The stable defense improvements for FedAvg are consistently over 1.5% times in LPIPS and SSIM. It also offers a universal defense against various GIA types, especially for these sensitive areas in images.

## Dataset

- Download datasets, i.e., [ChestXRay](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data) and [EyePACS](https://zenodo.org/records/5793241).
- Put datasets under 'data/Chest_X-ray' and 'data/EyePACS_AIROGS' folders in the same level directory of 'attack' and 'defense'.

## Task Training

cd ShadowDef/defense

Pretrain latent code z: 

- Change 'cfg.z_path' in '*_config.py' to None
- CUDA_VISIBLE_DEVICES=0 python train_main.py --config chestXray_config --gpu 1 --server_gpu 0 --exp_name client_z

FL training:

- Change 'cfg.z_path' in '*_config.py' to "../output_checkXray/client_z/"
- CUDA_VISIBLE_DEVICES=0 python train_main.py --config chestXray_config --gpu 1 --server_gpu 0 --exp_name your_exp_name

Note:

- Change 'chestXray_config ' to eye_config' to train on the EyePACS dataset.
- Create folder 'defense/net/GAN/pretrained_model' and download [pretrained weights](https://drive.google.com/drive/folders/1A6YmEbc8_DLYmKD6RDPlY3g7tkPtoIPB?usp=sharing) under the folder.

## Attack

Perform reconstruction

- Change your_path in code, '../data/Chest_X-ray' to '../../data/Chest_X-ray'
- sh attack_chestXRay.sh

Evaluation:

- Change your_path in code
- sh test.sh
- sh test_target_region.sh
