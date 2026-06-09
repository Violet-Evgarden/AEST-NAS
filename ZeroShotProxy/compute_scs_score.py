import math
import copy
import torch

from ZeroShotProxy import compute_zen_score
from ZeroShotProxy import compute_zico_score
from ZeroShotProxy import compute_syncflow_score


def compute_scs(model, resolution, batch_size, mixup_gamma=1e-2, gpu=0):

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    model_cpu = model.cpu()
    model_zico = copy.deepcopy(model_cpu).cuda(gpu)
    model_zen = copy.deepcopy(model_cpu).cuda(gpu)
    model_syn = copy.deepcopy(model_cpu).cuda(gpu)

    try:
        model.cuda(gpu)

        zico_val, zen_val, syn_val = 1e-4, 1e-4, 1e-4

        try:
            zico_val = compute_zico_score.compute_nas_score(model=model_zico, gpu=gpu, resolution=resolution,
                                                            batch_size=batch_size)
            if math.isnan(zico_val) or math.isinf(zico_val) or zico_val <= 0:
                zico_val = 1e-4
        except Exception as e:
            pass

        try:
            zen_val = compute_zen_score.compute_zen_score(model=model_zen, resolution=resolution, batch_size=batch_size,
                                                          mixup_gamma=mixup_gamma, repeat=1, gpu=gpu)
            if math.isnan(zen_val) or math.isinf(zen_val) or zen_val <= 0:
                zen_val = 1e-4
        except Exception as e:
            pass

        try:
            syn_val = compute_syncflow_score.do_compute_nas_score(model=model_syn, gpu=gpu, resolution=resolution,
                                                                  batch_size=batch_size)
            if math.isnan(syn_val) or math.isinf(syn_val) or syn_val <= 0:
                syn_val = 1e-4
        except Exception as e:
            pass

        if zico_val == 1e-4 and zen_val == 1e-4 and syn_val == 1e-4:
            return 0.0

        log_syn_val = math.log10(syn_val + 1.0)

        divergence = abs((zico_val / zen_val) - log_syn_val)

        scs_fitness = (zico_val * zen_val) / (divergence + 1.0)

        valid_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad and len(p.shape) > 1)
        capacity_multiplier = math.log10(max(valid_param_count, 10))

        ultimate_scs = scs_fitness * capacity_multiplier

        return float(ultimate_scs)

    finally:
        model.cuda(gpu)
        try:
            del model_zico, model_zen, model_syn
        except:
            pass
        torch.cuda.empty_cache()