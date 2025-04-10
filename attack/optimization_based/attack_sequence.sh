source activate fed
cd your_path/tmi

# ChestXray
# client 0
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client0_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client0_model/client0_grad_epoch1.pkl --client_id 0 --pretrained_size 4 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client0_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client0_model/client0_grad_epoch25.pkl --client_id 0 --pretrained_size 4 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client0_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client0_model/client0_grad_epoch50.pkl --client_id 0 --pretrained_size 4 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client0_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client0_model/client0_grad_epoch75.pkl --client_id 0 --pretrained_size 4 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client0_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client0_model/client0_grad_epoch100.pkl --client_id 0 --pretrained_size 4 3 224 224

# client 1
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client1_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client1_model/client1_grad_epoch1.pkl --client_id 1 --pretrained_size 4 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client1_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client1_model/client1_grad_epoch25.pkl --client_id 1 --pretrained_size 4 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client1_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client1_model/client1_grad_epoch50.pkl --client_id 1 --pretrained_size 4 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client1_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client1_model/client1_grad_epoch75.pkl --client_id 1 --pretrained_size 4 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client1_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client1_model/client1_grad_epoch100.pkl --client_id 1 --pretrained_size 4 3 224 224

# client 2
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client2_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client2_model/client2_grad_epoch1.pkl --client_id 2 --pretrained_size 4 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client2_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client2_model/client2_grad_epoch25.pkl --client_id 2 --pretrained_size 4 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client2_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client2_model/client2_grad_epoch50.pkl --client_id 2 --pretrained_size 4 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client2_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client2_model/client2_grad_epoch75.pkl --client_id 2 --pretrained_size 4 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client2_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client2_model/client2_grad_epoch100.pkl --client_id 2 --pretrained_size 4 3 224 224

# client 3
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client3_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client3_model/client3_grad_epoch1.pkl --client_id 3 --pretrained_size 4 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client3_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client3_model/client3_grad_epoch25.pkl --client_id 3 --pretrained_size 4 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client3_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client3_model/client3_grad_epoch50.pkl --client_id 3 --pretrained_size 4 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client3_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client3_model/client3_grad_epoch75.pkl --client_id 3 --pretrained_size 4 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client3_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client3_model/client3_grad_epoch100.pkl --client_id 3 --pretrained_size 4 3 224 224

# client 4
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client4_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client4_model/client4_grad_epoch1.pkl --client_id 4 --pretrained_size 8 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client4_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client4_model/client4_grad_epoch25.pkl --client_id 4 --pretrained_size 8 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client4_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client4_model/client4_grad_epoch50.pkl --client_id 4 --pretrained_size 8 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client4_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client4_model/client4_grad_epoch75.pkl --client_id 4 --pretrained_size 8 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client4_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client4_model/client4_grad_epoch100.pkl --client_id 4 --pretrained_size 8 3 224 224

# client 5
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client5_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client5_model/client5_grad_epoch1.pkl --client_id 5 --pretrained_size 8 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client5_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client5_model/client5_grad_epoch25.pkl --client_id 5 --pretrained_size 8 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client5_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client5_model/client5_grad_epoch50.pkl --client_id 5 --pretrained_size 8 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client5_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client5_model/client5_grad_epoch75.pkl --client_id 5 --pretrained_size 8 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client5_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client5_model/client5_grad_epoch100.pkl --client_id 5 --pretrained_size 8 3 224 224

# client 6
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client6_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client6_model/client6_grad_epoch1.pkl --client_id 6 --pretrained_size 8 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client6_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client6_model/client6_grad_epoch25.pkl --client_id 6 --pretrained_size 8 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client6_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client6_model/client6_grad_epoch50.pkl --client_id 6 --pretrained_size 8 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client6_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client6_model/client6_grad_epoch75.pkl --client_id 6 --pretrained_size 8 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client6_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client6_model/client6_grad_epoch100.pkl --client_id 6 --pretrained_size 8 3 224 224

# client 7
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client7_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client7_model/client7_grad_epoch1.pkl --client_id 7 --pretrained_size 8 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client7_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client7_model/client7_grad_epoch25.pkl --client_id 7 --pretrained_size 8 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client7_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client7_model/client7_grad_epoch50.pkl --client_id 7 --pretrained_size 8 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client7_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client7_model/client7_grad_epoch75.pkl --client_id 7 --pretrained_size 8 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client7_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client7_model/client7_grad_epoch100.pkl --client_id 7 --pretrained_size 8 3 224 224

# client 8
# 0%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client8_prec0 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch1.pkl --client_id 8 --pretrained_size 1 3 224 224

# 25%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client8_prec25 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch25.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch25.pkl --client_id 8 --pretrained_size 1 3 224 224

# 50%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client8_prec50 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch50.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch50.pkl --client_id 8 --pretrained_size 1 3 224 224

# 75%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client8_prec75 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch75.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch75.pkl --client_id 8 --pretrained_size 1 3 224 224

# 100%
CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/client8_prec100 --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch100.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch100.pkl --client_id 8 --pretrained_size 1 3 224 224

# # debug
# CUDA_VISIBLE_DEVICES=0 python gradinv.py -c eye_config --output_path ../output_attack/FedAvg/try --global_model_path your_path/output_eye/FedAvg/model/global_model_epoch1.pth --grad_path your_path/output_eye/FedAvg/client8_model/client8_grad_epoch1.pkl --client_id 8 --pretrained_size 1 3 224 224