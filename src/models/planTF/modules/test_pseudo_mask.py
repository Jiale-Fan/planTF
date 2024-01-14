
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def show_mask():
    mask_prob = torch.randn(1, 10)
    k=1
    z = torch.zeros_like(mask_prob)

    for _ in range(k):
        mask = F.gumbel_softmax(mask_prob, dim=1, tau=0.5, hard=False)
        z = torch.maximum(mask,z)

    z = z[0]
    # visualize z using histogram
        
    
    plt.bar(range(z.shape[0]), z.detach().numpy().flatten())
    plt.savefig('./debug_files/z.png')
    plt.close()


if __name__ == '__main__':
    for _ in range(10):
        show_mask()
        pass