# Rescale Vision Transformer
### 1. Introduction

Pytorch implementation of RescaleViT. The code for HKU COMP9501: Machine learing. Group members: Tao Chaofan and Deng weipeng.

Dictionary structure:
```
RescaleViT
   |--img # Some visualization files
   |--logs #  Tensorboard file during training
   |--models # Model design
   |--ouptut # Result log files during evaluation
   |--utils # Some utilization including data loader, distributed training, learning scheduler and visualizers
```



### 2. Train and evaluate Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --norm_type Rescale  --model_type ViT-B_16  --fp16 --fp16_opt_level O2
```
```--name``` is the name of experiment, which is alo the prefix of saved checkpoint name and log file name.

```--dataset``` is the dataset,  you can choose ```cifar10 ``` or ```cifar100 ```.

```--norm_type``` is the type of normalization, you can choose  ```Rescale``` (without normalization), or ```LN```, ```BN```, ```GN```.

```--model_type``` is the architecture setting, you can use ```ViT-B_16```, ```ViT-B_32```, ```ViT-L_16``` and ```ViT-L_32```

```--pretrained_dir``` [optional] is the path of pretrained file, which can be none (training from scratch).

```---fp16``` [optional] means use fp16 training powered by [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html), which reduces GPU memory usage **3~4x** during training.

We train the model with pretrained files as follows:
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

```

If you want to train all the architectures with different normalizations (take a long time for all experiments), you can simply use
```
sh train1.sh # for cifar10
sh train3.sh # for cifar100
```

<br>

## Results
We conduct all experiments on a single NVIDIA-3090 card. The default batch size is 128. We train 20000 iterations for all experiments. All the evaluation is made every ```---eval_every```(default setting is 100) iterations. And all the evaluation log files are available in ```"./output/"```. 

We use "*" denote the model initialized with pretained checkpoint. From the results, we can observe that the pretained checkpoint on the Imagenet21k plays a significant role in  training ViT, for various architectures. The model used pretained checkpoint outperform the CNN-based model. 

On the other hand, for the model trained from scratch, the results cannot beat CNN-based model, which is consists with the obseration in the paper [ViT](https://arxiv.org/abs/2010.11929).

Besides the past experience about CNN, we have several empirical deduction.
1. Transformer has potential to beat CNN-based model, while it generally needs more parameters and more training time. 
2. The pretained checkpoint is very important for training the transformer. It is understandable since transformer has much more parameters. For example, ViT-B_16(**85.8M**) v.s ResNet18 (**11.7M**).
3. Layer-normalization has minor improvements for transformer on the CIFAR10 and CIFAR100, compared to BN, GN. In addition, RescaleViT can achieve similar results. Both pretained model and RescaleViT can speedup the training time a little bit.

### CIFAR-10
* [**tensorboard**](./logs/) contains some of training information, we donot upload all the tensorboards file due to space limitation. "+" denotes the normalization method used in ViT.

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_16    |  + | | |                |  224x224   |      69.7          |  171.1 |
|   ViT-B_16*  |   + | | |             |  224x224   |        **98.4**      |  170.6 |
|   ViT-B_16   |  | + ||             |  224x224   |           66.5      |   181.1 |
|   ViT-B_16   |   | | +    |        |  224x224   |        65.1        |   185.1|
|   ViT-B_16   |    |  | | +           |  224x224   |        65.2        | **165.9**  |
|   ViT-B_32   |  + | | |                |  224x224   |      67.6          | 68.5  |
|   ViT-B_32*  |   + | | |             |  224x224   |        **98.4**      |   67.7 |
|   ViT-B_32   |  | + ||             |  224x224   |         67.1        |  68.6 |
|   ViT-B_32   |   | | +    |        |  224x224   |         65.5       |    73.0|
|   ViT-B_32   |    |  | | +           |  224x224   |         66.1       |  **66.7** |
|   ViT-L_16   |  + | | |                |  224x224   |                |   |
|   ViT-L_16*  |   + | | |             |  224x224   |       **99.0**       |  308.2 |
|   ViT-L_16   |  | + ||             |  224x224   |                 |   |
|   ViT-L_16   |   | | +    |        |  224x224   |                |   |
|   ViT-L_16   |    |  | | +           |  224x224   |                |   |
|   ViT-L_32   |  + | | |                |  224x224   |      67.5          |  146.2  |
|   ViT-L_32*  |   + | | |             |  224x224   |      **98.9**        | **144.3**   |
|   ViT-L_32   |  | + ||             |  224x224   |         67.4        | 152.5  |
|   ViT-L_32   |   | | +    |        |  224x224   |         65.6       |  163.3 |
|   ViT-L_32   |    |  | | +           |  224x224   |       65.4         |  145.8 |

<br>

### CIFAR-100
|    model     |  LN | BN | GN | Rescale  | resolution | acc |  time   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_16   |  + | | |                |  224x224   |       43.5        |  168.8 |
|   ViT-B_16*  |   + | | |             |  224x224   |         **91.9**     | 167.7   |
|   ViT-B_16   |  | + ||             |  224x224   |         38.7        |  177.5  |
|   ViT-B_16   |   | | +    |        |  224x224   |          38.2      |  181.4 |
|   ViT-B_16   |    |  | | +           |  224x224   |       36.1         | **162.6**  |
|   ViT-B_32   |  + | | |                |  224x224   |         42.8       |  68.4 |
|   ViT-B_32*  |   + | | |             |  224x224   |        **91.0**      | **65.6**  |
|   ViT-B_32   |  | + ||             |  224x224   |           39.9      |68.7    |
|   ViT-B_32   |   | | +    |        |  224x224   |          40.0      |  73.1 |
|   ViT-B_32   |    |  | | +           |  224x224   |         37.8       | 67.5  |
|   ViT-L_16   |  + | | |                |  224x224   |                |   |
|   ViT-L_16*  |   + | | |             |  224x224   |              |   |
|   ViT-L_16   |  | + ||             |  224x224   |                 |   |
|   ViT-L_16   |   | | +    |        |  224x224   |                |   |
|   ViT-L_16   |    |  | | +           |  224x224   |                |   |
|   ViT-L_32   |  + | | |                |  224x224   |   42.7           |  144.6 |
|   ViT-L_32*  |   + | | |             |  224x224   |    **92.2**         |  **143.4**    |
|   ViT-L_32   |  | + ||             |  224x224   |       40.6          |   150.2|
|   ViT-L_32   |   | | +    |        |  224x224   |       39.6         |   160.5 |
|   ViT-L_32   |    |  | | +           |  224x224   |       37.6         | 143.6  |

<br>

## Visualization
### Attention Map

The ViT consists of a Standard Transformer Encoder, and the encoder consists of Self-Attention and MLP module.
The attention map for the input image can be visualized through the attention score of self-attention.

Here is the learned attention map of ViT-B_16 on cifar10 dataset.
We visualize the attention map at fisrt head in 0, 3, 6 ,9 block.

We can observe that the attention map is sparsely distributed regardless of normalization methods. It is noteworthy that the pretrained LN-model learn the most important components along the diagonal, which means the learned attention is focus on each neuron itself.



LN
<div align="center">
<img src="./img/plot_attn/LN_layer_0.png" height="170sx" >
<img src="./img/plot_attn/LN_layer_3.png" height="170sx"  >
<img src="./img/plot_attn/LN_layer_6.png" height="170sx"  >
<img src="./img/plot_attn/LN_layer_9.png" height="170sx"  >
</div>

LN (pretrain)
<div align="center">
<img src="./img/plot_attn/LN2_layer_0.png" height="170sx" >
<img src="./img/plot_attn/LN2_layer_3.png" height="170sx"  >
<img src="./img/plot_attn/LN2_layer_6.png" height="170sx"  >
<img src="./img/plot_attn/LN2_layer_9.png" height="170sx"  >
</div>

BN
<div align="center">
<img src="./img/plot_attn/BN_layer_0.png" height="170sx" >
<img src="./img/plot_attn/BN_layer_3.png" height="170sx"  >
<img src="./img/plot_attn/BN_layer_6.png" height="170sx"  >
<img src="./img/plot_attn/BN_layer_9.png" height="170sx"  >
</div>

GN
<div align="center">
<img src="./img/plot_attn/GN_layer_0.png" height="170sx" >
<img src="./img/plot_attn/GN_layer_3.png" height="170sx"  >
<img src="./img/plot_attn/GN_layer_6.png" height="170sx"  >
<img src="./img/plot_attn/GN_layer_9.png" height="170sx"  >
</div>

Rescale (without normalization)
<div align="center">
<img src="./img/plot_attn/Rescale_layer_0.png" height="170sx" >
<img src="./img/plot_attn/Rescale_layer_3.png" height="170sx"  >
<img src="./img/plot_attn/Rescale_layer_6.png" height="170sx"  >
<img src="./img/plot_attn/Rescale_layer_9.png" height="170sx"  >
</div>

<br>


### Loss and Accuracy
The validation loss and accuracy are illustrated below. Architecture: ViT-B_32, Dataset:CIFAR-10

LN and LN*(pretrained on ImageNet21k) 
<div align="center">
<img src="./img/plot_acc_loss/LN1_loss_acc.png" height="250sx" >
<img src="./img/plot_acc_loss/LN2_loss_acc.png" height="250sx"  >
</div>

BN and GN
<div align="center">
<img src="./img/plot_acc_loss/BN_loss_acc.png" height="250sx" >
<img src="./img/plot_acc_loss/GN_loss_acc.png" height="250sx"  >
</div>

Rescale (without normalization)
<div align="center">
<img src="./img/plot_acc_loss/Rescale_loss_acc.png" height="250sx" >
</div>
<br>

## Acknowlegement
The code is implemented based on [ViT](https://github.com/jeonsworld/ViT-pytorch).

