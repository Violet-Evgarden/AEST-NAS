import os, sys
import torch
from torch import nn
import numpy as np


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    return net


def compute_zen_score(model, resolution, batch_size, mixup_gamma=1e-2, repeat=32, gpu=0):
    model.train()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')

    nas_score_list = []

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)


            input_x = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
            input_y = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
            mixup_input = input_x + mixup_gamma * input_y

            out_x = model(input_x)
            if isinstance(out_x, tuple):
                out_x = out_x[0]

            out_mix = model(mixup_input)
            if isinstance(out_mix, tuple):
                out_mix = out_mix[0]

            diff = torch.abs(out_x - out_mix)

            nas_score = torch.mean(torch.sum(diff, dim=1))

            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'running_var') and m.running_var is not None:
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var) + 1e-5)
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            nas_score = torch.log(nas_score + 1e-5) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score