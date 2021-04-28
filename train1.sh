## cifar10
# train on ViT,  LN

# # ------- train fp16, imgsize 224
# arch_list="ViT-B_16 ViT-B_32 ViT-L_16 ViT-L_32"  
# # train on ViT, LN
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch  --fp16 --fp16_opt_level O2;
# done 

# # train on ViT, BN
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch  --fp16 --fp16_opt_level O2;
# done 




# arch_list="ViT-B_16 ViT-B_32 ViT-L_16 ViT-L_32"  
# # train on ViT, GN
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch  --fp16 --fp16_opt_level O2;
# done 

# # train on RescaleViT (with fp16 training)
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch  --fp16 --fp16_opt_level O2;
# done 


# ------- train both fp32 and fp32-16 mixed, imgsize 128
# arch_list="ViT-B_16 ViT-B_32 ViT-L_16 ViT-L_32"  
# # train on ViT, LN
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch;
# done 

# arch_list="ViT-B_16 ViT-B_32 ViT-L_32"  
# for arch in $arch_list;  
# do  
# python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --pretrained_dir ./pretrain/$arch.npz --fp16 --fp16_opt_level O2;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
# python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
# done 


# due to memory limit, we train ViT-L_16 with batch size 64
arch=ViT-L_16
# python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --pretrained_dir ./pretrain/$arch.npz --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64; 


