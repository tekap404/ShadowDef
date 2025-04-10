source activate fed
cd your_path/tmi

# eye
# client 0
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client0_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 0  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client0_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 0  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client0_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 0  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client0_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 0  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client0_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 0  > ./5.log 2>&1 &

# client 1
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client1_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 1  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client1_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 1  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client1_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 1  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client1_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 1  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client1_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 1  > ./5.log 2>&1 &

# client 2
CUDA_VISIBLE_DEVICES=3 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client2_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 2  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client2_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 2  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client2_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 2  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client2_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 2  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client2_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 2  > ./5.log 2>&1 &

# client 3
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client3_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 3  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client3_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 3  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client3_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 3  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client3_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 3  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client3_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 4 3 224 244 --client_id 3  > ./5.log 2>&1 &

# client 4
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client4_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 4  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client4_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 4  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client4_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 4  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client4_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 4  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client4_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 4  > ./5.log 2>&1 &

# client 5
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client5_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 5  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client5_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 5  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client5_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 5  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client5_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 5  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client5_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 5  > ./5.log 2>&1 &

# client 6
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client6_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 6  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client6_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 6  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client6_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 6  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client6_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 6  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client6_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 6  > ./5.log 2>&1 &

# client 7
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client7_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 7  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client7_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 7  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client7_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 7  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client7_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 7  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client7_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 8 3 224 244 --client_id 7  > ./5.log 2>&1 &

# client 8
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client8_prec0 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 1 3 224 244 --client_id 8  > ./1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client8_prec25 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 1 3 224 244 --client_id 8  > ./2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client8_prec50 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 1 3 224 244 --client_id 8  > ./3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client8_prec75 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 1 3 224 244 --client_id 8  > ./4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_target_region.py -c eye_config --output_path ../output_attack/eye/FedAvg/client8_prec100 --global_model_path your_path/output_eye/FedAvg/model/best_global_model.pth --cal_target True  --pretrained_size 1 3 224 244 --client_id 8  > ./5.log 2>&1 &

