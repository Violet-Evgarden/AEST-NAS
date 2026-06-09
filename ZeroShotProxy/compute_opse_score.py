import os, sys
import torch
import numpy as np

from ZeroShotProxy import compute_zen_score
from ZeroShotProxy import compute_zico_score
from ZeroShotProxy import compute_syncflow_score


ZICO_GLOBAL_MEAN = 450.0
ZICO_GLOBAL_STD = 180.0

ZEN_GLOBAL_MEAN = 175.0
ZEN_GLOBAL_STD = 35.0

SYN_LOG_GLOBAL_MEAN = 0.5
SYN_LOG_GLOBAL_STD = 1.5


def compute_opse(model, resolution, batch_size, mixup_gamma=1e-2, gpu=0):

    try:
        zico = compute_zico_score.compute_nas_score(gpu, model, resolution, batch_size)
    except:
        zico = 0.0

    try:
        zen = compute_zen_score.compute_zen_score(model, resolution, batch_size, mixup_gamma, repeat=1, gpu=gpu)
    except:
        zen = 0.0

    try:
        synflow = compute_syncflow_score.do_compute_nas_score(gpu, model, resolution, batch_size)
    except:
        synflow = 0.0

    z_zico = (zico - ZICO_GLOBAL_MEAN) / ZICO_GLOBAL_STD
    z_zen = (zen - ZEN_GLOBAL_MEAN) / ZEN_GLOBAL_STD
    z_syn = (np.log(synflow + 1e-8) - SYN_LOG_GLOBAL_MEAN) / SYN_LOG_GLOBAL_STD

    proxy_scores = [z_zico, z_zen, z_syn]

    mu = np.mean(proxy_scores)
    sigma = np.std(proxy_scores)

    alpha = 1.5

    opse_final_score = mu - alpha * sigma

    return float(opse_final_score)