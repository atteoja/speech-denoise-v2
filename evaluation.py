import torch
import torch.nn as nn


def get_psnr(pred, gt, max_value=1.0):
    mse = torch.mean(torch.pow(pred - gt, 2))
    return 20 * torch.log10(max_value / torch.sqrt(mse))