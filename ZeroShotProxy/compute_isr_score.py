import torch
from torch import nn
import numpy as np


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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
def compute_isr(model, resolution, batch_size, mixup_gamma=1e-2, repeat=32, gpu=0):
    model.train()
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    if gpu is not None:
        model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    nas_score_list = []
    grad_dict = {}
    for step in range(repeat):
        network_weight_gaussian_init(model)
        model.zero_grad()

        input_x = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
        input_y = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
        mixup_input = input_x + mixup_gamma * input_y

        out_x = model(input_x)
        out_x = out_x[0] if isinstance(out_x, tuple) else out_x

        out_mix = model(mixup_input)
        out_mix = out_mix[0] if isinstance(out_mix, tuple) else out_mix

        diff = torch.abs(out_x - out_mix)
        nas_score = torch.mean(torch.sum(diff, dim=1))

        log_bn_scaling_factor = 0.0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'running_var') and m.running_var is not None:
                bn_scaling_factor = torch.sqrt(torch.mean(m.running_var) + 1e-5)
                log_bn_scaling_factor += torch.log(bn_scaling_factor)

        expressivity = torch.log(nas_score + 1e-5) + log_bn_scaling_factor
        nas_score_list.append(float(expressivity))

        loss = torch.sum(diff)
        loss.backward()

        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if layer.weight.grad is not None:
                    grad_val = layer.weight.grad.data.cpu().reshape(-1).numpy()
                    if name not in grad_dict:
                        grad_dict[name] = []
                    grad_dict[name].append(grad_val)

    final_zen = float(np.mean(nas_score_list))
    if final_zen <= 0:
        final_zen = 1e-5  #

    layer_snrs = []
    for name, grads in grad_dict.items():
        grads_arr = np.array(grads)

        grad_mean = np.mean(np.abs(grads_arr), axis=0)
        grad_std = np.std(grads_arr, axis=0)

        valid_idx = np.nonzero(grad_std)[0]
        if len(valid_idx) > 0:
            snr = grad_mean[valid_idx] / grad_std[valid_idx]
            layer_snrs.append(np.mean(snr))

    if len(layer_snrs) == 0:
        return 0.0

    layer_snrs = np.array(layer_snrs)

    mu_snr = np.mean(layer_snrs)

    sigma_snr = np.std(layer_snrs)

    isr_fitness = final_zen * (mu_snr / (sigma_snr + 1e-4))

    return float(isr_fitness)