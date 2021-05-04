# Rescale Vision Transformer
### 1. Introduction

Pytorch implementation of RescaleViT. 

The project investigates vision transformer with **different normalization methods (LN, GN, BN)**, or **training without normalization and without performance degradation**. [[PDF]](https://github.com/ChaofanTao/RescaleViT/blob/master/RescaleViT%20Training%20Vision%20Transformer%20withoutNormalization.pdf)

**HKU-COMP9501 course project**. Group members: Tao Chaofan and Deng weipeng. 

Directory structure:
```
RescaleViT
   |--/img # Some visualization files
   |--/logs #  Tensorboard file during training
   |--/models # Model design
   |--/ouptut # Result log files during evaluation
   |--/utils # Some utilization including data loader, distributed training, learning scheduler and visualizers
   train.py # the main program
   train1.sh # script to quickly start all the experiments on cifar10
   train3.sh # script to quickly start all the experiments on cifar100
```



### 2. Train and evaluate Model
```
pip install -r requirements.txt

python3 train.py --name cifar10-100_500 --dataset cifar10 \ 
                 --norm_type Rescale  --model_type ViT-B_16 \
                 --fp16 --fp16_opt_level O2
```
```--name``` is the name of experiment, which is also the prefix of saved checkpoint name and log file name.

```--dataset``` is the dataset,  you can choose ```cifar10 ``` or ```cifar100 ```.

```--norm_type``` is the type of normalization, you can choose  ```Rescale``` (without normalization), or ```LN```, ```BN```, ```GN```. The idea of using ```Rescale``` to replace normalization is firstly adopted by [Is normalization indispensable for training deep neural networks?](https://papers.nips.cc/paper/2020/file/9b8619251a19057cff70779273e95aa6-Paper.pdf) on the CNNs.

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

If you want to train all the architectures with different normalizations (take 2 days+  for all experiments), you can simply use
```
sh train1.sh # for cifar10
sh train3.sh # for cifar100
```

<br>

## Results
We conduct all single experiment on one NVIDIA-3090 card. The default batch size is 128. We train 20000 iterations for all experiments. All the evaluation is made every ```---eval_every```(default setting is 100) iterations. And all the evaluation log files are available in ```"./output/"```. 

From the results, we can observe that the pretained checkpoint on the Imagenet21k plays a significant role in  training ViT, for various architectures. The model used pretained checkpoint outperform the CNN-based model. 

On the other hand, for the model trained from scratch, the results cannot beat CNN-based model, which is consists with the obseration in the paper [ViT](https://arxiv.org/abs/2010.11929).

Besides the past experience about CNN, we have several empirical deductions.
1. Transformer has potential to outperform CNN-based model with a large margin, while it generally needs more data, more parameters and more training time. 
2. The pretained checkpoint on Imagenet21k is very important for training the transformer on CIFAR-10 and CIFAR-100. It is understandable since transformer has much more parameters. For example, ViT-B_16(**85.8M**) v.s ResNet18 (**11.7M**).
3. Generally speaking, LN > BN ~= Rescale > GN on the CIFAR10 and CIFAR100 in the performance.
4. Both pretained model and RescaleViT can speedup the training time a little bit.

### CIFAR-10
"+" denotes the normalization method used in ViT. We use "*" denote the model initialized with pretained checkpoint on Imagenet21k. We train the model using fp16 precision.


|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_16    |  + | | |                |  224x224   |      69.7          |  171.1 |
|   ViT-B_16*  |   + | | |             |  224x224   |        **98.4**      |  170.6 |
|   ViT-B_16   |  | + ||             |  224x224   |           66.5      |   181.1 |
|   ViT-B_16   |   | | +    |        |  224x224   |        65.1        |   185.1|
|   ViT-B_16   |    |  | | +           |  224x224   |        65.2        | **165.9**  |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_32   |  + | | |                |  224x224   |      67.6          | 68.5  |
|   ViT-B_32*  |   + | | |             |  224x224   |        **98.4**      |   67.7 |
|   ViT-B_32   |  | + ||             |  224x224   |         67.1        |  68.6 |
|   ViT-B_32   |   | | +    |        |  224x224   |         65.5       |    73.0|
|   ViT-B_32   |    |  | | +           |  224x224   |         66.1       |  **66.7** |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-L_16   |  + | | |                |  224x224   |      66.3          |  306.2 |
|   ViT-L_16*  |   + | | |             |  224x224   |       **99.0**       |  308.2 |
|   ViT-L_16   |  | + ||             |  224x224   |       63.0          |  326.4 |
|   ViT-L_16   |   | | +    |        |  224x224   |         60.8       |  330.9 |
|   ViT-L_16   |    |  | | +           |  224x224   |        62.9        |  **295.0**  |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-L_32   |  + | | |                |  224x224   |      67.5          |  146.2  |
|   ViT-L_32*  |   + | | |             |  224x224   |      **98.9**        | **144.3**   |
|   ViT-L_32   |  | + ||             |  224x224   |         67.4        | 152.5  |
|   ViT-L_32   |   | | +    |        |  224x224   |         65.6       |  163.3 |
|   ViT-L_32   |    |  | | +           |  224x224   |       65.4         |  145.8 |

<br>

### CIFAR-100
|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_16   |  + | | |                |  224x224   |       43.5        |  168.8 |
|   ViT-B_16*  |   + | | |             |  224x224   |         **91.9**     | 167.7   |
|   ViT-B_16   |  | + ||             |  224x224   |         38.7        |  177.5  |
|   ViT-B_16   |   | | +    |        |  224x224   |          38.2      |  181.4 |
|   ViT-B_16   |    |  | | +           |  224x224   |       36.1         | **162.6**  |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-B_32   |  + | | |                |  224x224   |         42.8       |  68.4 |
|   ViT-B_32*  |   + | | |             |  224x224   |        **91.0**      | **65.6**  |
|   ViT-B_32   |  | + ||             |  224x224   |           39.9      |68.7    |
|   ViT-B_32   |   | | +    |        |  224x224   |          40.0      |  73.1 |
|   ViT-B_32   |    |  | | +           |  224x224   |         37.8       | 67.5  |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
|   ViT-L_16   |  + | | |                |  224x224   |      40.2          |  303.9 |
|   ViT-L_16*  |   + | | |             |  224x224   |        **93.2**      |   298.8 |
|   ViT-L_16   |  | + ||             |  224x224   |         34.5        |  320.5 |
|   ViT-L_16   |   | | +    |        |  224x224   |        31.6        |  326.7  |
|   ViT-L_16   |    |  | | +           |  224x224   |         34.6       |   **295.6** |

|    model     |  LN | BN | GN | Rescale  | resolution | acc(%) |  time(min)   |
|:------------:|:--:|:--:|:--:|:------:|:-------------:|:-----:|:-------:|
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
We visualize the attention map at fisrt head in **0, 3rd, 6th ,9th block**.

We can observe that the attention map is sparsely distributed regardless of normalization methods. It is noteworthy that the pretrained LN-model learn the most important components along the diagonal, which means the learned attention is focus on each neuron itself. RescaleViT has more sparse attention than the model using normalization.



LN
<div align="center">
<img src="./img/plot_attn/LN_layer_0.png" height="150sx" >
<img src="./img/plot_attn/LN_layer_3.png" height="150sx"  >
<img src="./img/plot_attn/LN_layer_6.png" height="150sx"  >
<img src="./img/plot_attn/LN_layer_9.png" height="150sx"  >
</div>

LN (pretrain)
<div align="center">
<img src="./img/plot_attn/LN2_layer_0.png" height="150sx" >
<img src="./img/plot_attn/LN2_layer_3.png" height="150sx"  >
<img src="./img/plot_attn/LN2_layer_6.png" height="150sx"  >
<img src="./img/plot_attn/LN2_layer_9.png" height="150sx"  >
</div>

BN
<div align="center">
<img src="./img/plot_attn/BN_layer_0.png" height="150sx" >
<img src="./img/plot_attn/BN_layer_3.png" height="150sx"  >
<img src="./img/plot_attn/BN_layer_6.png" height="150sx"  >
<img src="./img/plot_attn/BN_layer_9.png" height="150sx"  >
</div>

GN
<div align="center">
<img src="./img/plot_attn/GN_layer_0.png" height="150sx" >
<img src="./img/plot_attn/GN_layer_3.png" height="150sx"  >
<img src="./img/plot_attn/GN_layer_6.png" height="150sx"  >
<img src="./img/plot_attn/GN_layer_9.png" height="150sx"  >
</div>

Rescale (without normalization)
<div align="center">
<img src="./img/plot_attn/Rescale_layer_0.png" height="150sx" >
<img src="./img/plot_attn/Rescale_layer_3.png" height="150sx"  >
<img src="./img/plot_attn/Rescale_layer_6.png" height="150sx"  >
<img src="./img/plot_attn/Rescale_layer_9.png" height="150sx"  >
</div>

<br>


### Loss and Accuracy
The validation loss and accuracy are illustrated below. Architecture: ViT-B_32, Dataset:CIFAR-10. 

Green line is loss curve and blue line is accuracy curve. The model with (LN+pretained) has a good initialization that even achieve over 90% accuracy before training, and it converges quickly. 

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

