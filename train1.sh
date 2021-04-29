# cifar10
arch_list="ViT-B_16 ViT-B_32 ViT-L_32"  
for arch in $arch_list;  
do  
python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --pretrained_dir ./pretrain/$arch.npz --fp16 --fp16_opt_level O2;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2;
done 


# due to memory limit, we train ViT-L_16 with batch size 64
arch=ViT-L_16
python3 train.py --gpu_id 0 --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --pretrained_dir ./pretrain/$arch.npz --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type LN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type BN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type GN --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64;
python3 train.py --gpu_id 0  --name cifar10-100_500 --dataset cifar10 --norm_type Rescale --model_type $arch --learning_rate 0.05 --fp16 --fp16_opt_level O2 --train_batch_size 64; 


