source activate fed
cd your_path/tmi

# ChestXray
# client 0
# 0%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client0_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client0_model/client0_grad_epoch1.pkl --client_id 0 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client0_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client0_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client0_model/client0_grad_epoch25.pkl --client_id 0 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client0_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client0_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client0_model/client0_grad_epoch50.pkl --client_id 0 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client0_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client0_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client0_model/client0_grad_epoch75.pkl --client_id 0 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client0_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client0_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client0_model/client0_grad_epoch100.pkl --client_id 0 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client0_perc100.log 2>&1 &

# client 1
# 0%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client1_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client1_model/client1_grad_epoch1.pkl --client_id 1 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client1_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client1_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client1_model/client1_grad_epoch25.pkl --client_id 1 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client1_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client1_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client1_model/client1_grad_epoch50.pkl --client_id 1 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client1_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client1_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client1_model/client1_grad_epoch75.pkl --client_id 1 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client1_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client1_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client1_model/client1_grad_epoch100.pkl --client_id 1 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client1_perc100.log 2>&1 &

# client 2
# 0%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client2_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client2_model/client2_grad_epoch1.pkl --client_id 2 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client2_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client2_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client2_model/client2_grad_epoch25.pkl --client_id 2 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client2_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client2_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client2_model/client2_grad_epoch50.pkl --client_id 2 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client2_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client2_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client2_model/client2_grad_epoch75.pkl --client_id 2 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client2_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client2_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client2_model/client2_grad_epoch100.pkl --client_id 2 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client2_perc100.log 2>&1 &

# client 3
# 0%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client3_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client3_model/client3_grad_epoch1.pkl --client_id 3 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client3_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client3_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client3_model/client3_grad_epoch25.pkl --client_id 3 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client3_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client3_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client3_model/client3_grad_epoch50.pkl --client_id 3 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client3_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client3_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client3_model/client3_grad_epoch75.pkl --client_id 3 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client3_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client3_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client3_model/client3_grad_epoch100.pkl --client_id 3 --pretrained_size 4 3 224 224 > ./chestXray_FedAvg_client3_perc100.log 2>&1 &
################
# client 4
# 0%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client4_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client4_model/client4_grad_epoch1.pkl --client_id 4 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client4_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client4_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client4_model/client4_grad_epoch25.pkl --client_id 4 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client4_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client4_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client4_model/client4_grad_epoch50.pkl --client_id 4 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client4_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client4_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client4_model/client4_grad_epoch75.pkl --client_id 4 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client4_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client4_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client4_model/client4_grad_epoch100.pkl --client_id 4 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client4_perc100.log 2>&1 &

# client 5
# 0%
CUDA_VISIBLE_DEVICES=2 nohup python gradinv.py --output_path ../output_attack/client5_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client5_model/client5_grad_epoch1.pkl --client_id 5 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client5_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=2 nohup python gradinv.py --output_path ../output_attack/client5_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client5_model/client5_grad_epoch25.pkl --client_id 5 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client5_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=2 nohup python gradinv.py --output_path ../output_attack/client5_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client5_model/client5_grad_epoch50.pkl --client_id 5 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client5_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=2 nohup python gradinv.py --output_path ../output_attack/client5_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client5_model/client5_grad_epoch75.pkl --client_id 5 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client5_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=2 nohup python gradinv.py --output_path ../output_attack/client5_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client5_model/client5_grad_epoch100.pkl --client_id 5 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client5_perc100.log 2>&1 &

# client 6
# 0%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client6_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client6_model/client6_grad_epoch1.pkl --client_id 6 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client6_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client6_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client6_model/client6_grad_epoch25.pkl --client_id 6 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client6_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client6_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client6_model/client6_grad_epoch50.pkl --client_id 6 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client6_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client6_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client6_model/client6_grad_epoch75.pkl --client_id 6 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client6_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client6_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client6_model/client6_grad_epoch100.pkl --client_id 6 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client6_perc100.log 2>&1 &

# client 7
# 0%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client7_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client7_model/client7_grad_epoch1.pkl --client_id 7 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client7_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client7_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client7_model/client7_grad_epoch25.pkl --client_id 7 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client7_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client7_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client7_model/client7_grad_epoch50.pkl --client_id 7 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client7_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client7_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client7_model/client7_grad_epoch75.pkl --client_id 7 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client7_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=0 nohup python gradinv.py --output_path ../output_attack/client7_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client7_model/client7_grad_epoch100.pkl --client_id 7 --pretrained_size 8 3 224 224 > ./chestXray_FedAvg_client7_perc100.log 2>&1 &

# client 8
# 0%
CUDA_VISIBLE_DEVICES=7 nohup python gradinv.py --output_path ../output_attack/client8_prec0 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch1.pkl --client_id 8 --pretrained_size 1 3 224 224 > ./chestXray_FedAvg_client8_perc0.log 2>&1 &

# 25%
CUDA_VISIBLE_DEVICES=7 nohup python gradinv.py --output_path ../output_attack/client8_prec25 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch25.pkl --client_id 8 --pretrained_size 1 3 224 224 > ./chestXray_FedAvg_client8_perc25.log 2>&1 &

# 50%
CUDA_VISIBLE_DEVICES=3 nohup python gradinv.py --output_path ../output_attack/client8_prec50 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch50.pkl --client_id 8 --pretrained_size 1 3 224 224 > ./chestXray_FedAvg_client8_perc50.log 2>&1 &

# 75%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client8_prec75 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch75.pkl --client_id 8 --pretrained_size 1 3 224 224 > ./chestXray_FedAvg_client8_perc75.log 2>&1 &

# 100%
CUDA_VISIBLE_DEVICES=5 nohup python gradinv.py --output_path ../output_attack/client8_prec100 --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch100.pkl --client_id 8 --pretrained_size 1 3 224 224 > ./chestXray_FedAvg_client8_perc100.log 2>&1 &

# debug
CUDA_VISIBLE_DEVICES=1 python gradinv.py --output_path ../output_attack/try --global_model_path your_path/output_checkXray/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_checkXray/FedAvg/client8_model/client8_grad_epoch1.pkl --client_id 8 --pretrained_size 1 3 224 224