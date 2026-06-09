import torch
from torch import nn
import numpy as np

def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict

def caculate_zico(grad_dict):
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    
    nsr_mean_sum_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            
    return nsr_mean_sum_abs

def compute_nas_score(gpu, model, resolution, batch_size):
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    model.train()
    lossfunc = nn.CrossEntropyLoss()
    grad_dict = {}


    num_batches = 32


    for i in range(num_batches):
        model.zero_grad()

        data = torch.randn(batch_size, 3, resolution, resolution)
        if gpu is not None:
            data = data.cuda(gpu)

        logits = model(data)

        if isinstance(logits, tuple):
            logits = logits[0]

        num_classes = logits.shape[1]
        label = torch.randint(0, num_classes, (batch_size,))
        if gpu is not None:
            label = label.cuda(gpu)

        loss = lossfunc(logits, label)
        loss.backward()

        grad_dict = getgrad(model, grad_dict, i)

    try:
        res = caculate_zico(grad_dict)
    except Exception as e:
        print(f"ZiCo  {e}")
        res = -9999.0

    model.zero_grad()
    torch.cuda.empty_cache()
    
    return float(res)