import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)


def vis_attn(args, attn_weights, layer_id=0):
    """
    attn_weights is selected attention maps in different layers. 
    attn_weights [len_h, len_h]
    """   
    save_dir = "./plot_attn"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    assert len(attn_weights.shape) == 2
    attn_weights = attn_weights.cpu().numpy()
    plt.imshow(attn_weights, cmap="Reds")
    savename = os.path.join(save_dir, "{}_layer_{}.png".format(args.norm_type, layer_id))
    plt.savefig(savename, dpi=500)
    print("save attn map to", savename)
    return


def vis_acc_loss(args, filepath):
    save_dir = "./plot_acc_loss"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    accs, losses = [], []
    print(filepath)

    with open(filepath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            if "Valid Loss" in line:
                loss = float(line.split(":")[-1].split("\n")[0])
                losses.append(loss)
            elif " Valid Accuracy" in line:
                acc = float(line.split(":")[-1].split("\n")[0])
                accs.append(acc)
    # import pdb;pdb.set_trace()
    fig = plt.figure()
    x = np.linspace(0, 20000, 200)

    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()          
    ax1.plot(x, losses,'g-')
    ax2.plot(x, accs,'b--')
    
    ax1.set_xlabel('Iteration')    
    ax1.set_ylabel('Loss',color = 'g')   
    ax2.set_ylabel('Accuracy',color = 'b')

    savename = os.path.join(save_dir, "{}_loss_acc.png".format(args.norm_type))
    plt.savefig(savename, dpi=500)
    print("save loss to", savename)
    return    





