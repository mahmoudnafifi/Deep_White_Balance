"""
 Loss function
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch


class mae_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss
