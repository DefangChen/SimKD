from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""
    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')
        
    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz,-1).mean(-1)

        loss = (weight * ind_loss).sum()/(1.0*bsz*num_stu)
        return loss