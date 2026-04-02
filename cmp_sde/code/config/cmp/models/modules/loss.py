import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys
import torch.nn as nn
from . import pytorch_ssim
import lpips

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted
        self.loss_f1 = None
        self.loss_f2 = None
        self.loss_fs = None
        self.loss_fl = None
        self.loss_type = loss_type
        print('loss_type=======',loss_type)
        if 'l1' in loss_type :
            self.loss_f1 = F.l1_loss # nn.L1Loss() # 
        if 'l2' in loss_type :
            self.loss_f2  =  F.mse_loss #  nn.MSELoss()  #
        if 'ssim' in loss_type :
            self.loss_fs = pytorch_ssim.SSIM()
        if 'lpips' in loss_type :
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.loss_fl = lpips.LPIPS(net='alex').to(device)
        # else:
        # raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):
        loss =0.0
        if self.loss_f1 is not None:
            loss_l1 =  self.loss_f1(predict, target,reduction='none')
            loss_l1 = einops.reduce(loss_l1, 'b ... -> b 1', 'mean')
            loss = loss +1*loss_l1.mean()
            # print('*****loss_l1:', loss_l1)
            # print('loss_l1',loss_l1.mean())
        
        
        if self.loss_f2 is not None:
            loss_l2 =  self.loss_f2(predict, target,reduction='none')
            loss_l2 = einops.reduce(loss_l2, 'b ... -> b (...)', 'mean')
            loss = loss +loss_l2.mean()
            # print('++++++loss_l2:', loss_l2.mean())
        if self.loss_fs is not None:
            loss_ls = 1 * (1 - self.loss_fs(predict, target)) 
            loss_ls = einops.reduce(loss_ls, 'b ... -> b (...)', 'mean')
            # print('loss_ls',loss_ls.mean())
            loss = loss +loss_ls.mean()
        if self.loss_fl is not None:
            loss_ll =  self.loss_fl(predict, target)
            loss_ll = einops.reduce(loss_ll, 'b ... -> b 1', 'mean')
            # print('=====loss_ll:', loss_ll.mean())
            loss = loss + 8*loss_ll.mean()

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss


