import os, sys, time
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


def get_layer_metric_array(net, metric, mode):
    metric_array = []
    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    return metric_array


def compute_synflow_per_weight(net, inputs, mode):
    device = inputs.device


    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.named_parameters():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs


    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.named_parameters():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)


    net.zero_grad()
    net.double()


    inputs = torch.ones_like(inputs).double().to(device)


    output = net(inputs)
    if isinstance(output, tuple):
        output = output[0]

    torch.sum(output).backward()


    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)


    nonlinearize(net, signs)


    net.float()

    return grads_abs


def do_compute_nas_score(gpu, model, resolution, batch_size):
    model.train()


    for param in model.parameters():
        param.requires_grad = True

    model.zero_grad()

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    network_weight_gaussian_init(model)
    input = torch.randn(size=[batch_size, 3, resolution, resolution])
    if gpu is not None:
        input = input.cuda(gpu)

    grads_abs_list = compute_synflow_per_weight(net=model, inputs=input, mode='')

    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))
        else:
            raise RuntimeError('!')

    return score