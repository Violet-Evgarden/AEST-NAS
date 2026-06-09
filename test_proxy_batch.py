import os
import sys
import torch
import torch.nn as nn
import argparse

# Dynamically add the current directory to the environment variables
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net


# 🌟 Bulletproof wrapper: Prevents tuple errors during derivation of various algorithms
class NAS201Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def test_batch_proxies():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID')

    # 🌟 One-click switch for the evaluation algorithm to test
    parser.add_argument('--proxy', type=str, default='Syncflow',
                        choices=['SCS', 'ISR', 'ZES', 'Zen', 'ZiCo', 'TE-NAS', 'Syncflow', 'GradNorm', 'NASWOT'],
                        help='Name of the evaluation algorithm to test')

    # 🌟 One-click switch for datasets
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Dataset to test')

    args = parser.parse_args()

    # Automatically match image size
    input_image_size = 16 if args.dataset == 'ImageNet16-120' else 32
    batch_size = 32

    print("Loading NAS-Bench-201 database...")
    nas201_api = API('NAS-Bench-201-v1_1-096897.pth')


    candidate_archs = [

]

    print(f"\n🚀 Starting batch validation | Metric: {args.proxy} | Dataset: {args.dataset}")
    print("=" * 160)

    # ==========================================
    # 🛠️ Dynamic module import section
    # ==========================================
    if args.proxy == 'SCS':
        from ZeroShotProxy import compute_scs_score
    elif args.proxy == 'ISR':
        from ZeroShotProxy import compute_isr_score
    elif args.proxy == 'ZiCo':
        from ZeroShotProxy import compute_zico_score
    elif args.proxy == 'Zen':
        from ZeroShotProxy import compute_zen_score
    elif args.proxy == 'TE-NAS':
        from ZeroShotProxy import compute_te_nas_score
    elif args.proxy == 'Syncflow':
        from ZeroShotProxy import compute_syncflow_score
    elif args.proxy == 'GradNorm':
        from ZeroShotProxy import compute_gradnorm_score
    elif args.proxy == 'NASWOT':
        from ZeroShotProxy import compute_NASWOT_score

    results_raw = []

    for item in candidate_archs:
        arch = item['arch']
        name = f"[{item.get('group', 'X')}] {item.get('name', 'Unknown')}"

        print(f"▶ Evaluating: {name}")
        try:
            arch_index = nas201_api.query_index_by_arch(arch)

            # ==========================================
            # 🛡️ Smart extraction of actual Valid and Test accuracies
            # ==========================================
            try:
                if args.dataset == 'cifar10':
                    # Valid and test for CIFAR-10 are stored separately in the API
                    info_test = nas201_api.get_more_info(arch_index, 'cifar10', hp='200', is_random=False)
                    info_valid = nas201_api.get_more_info(arch_index, 'cifar10-valid', hp='200', is_random=False)
                    test_acc = info_test['test-accuracy']
                    valid_acc = info_valid['valid-accuracy']
                else:
                    # Valid and test for CIFAR-100 and ImageNet16-120 are stored together
                    info = nas201_api.get_more_info(arch_index, args.dataset, hp='200', is_random=False)
                    test_acc = info['test-accuracy']
                    valid_acc = info['valid-accuracy']
            except Exception as e:
                print(f"  ⚠️ Failed to retrieve accuracy from table: {e}")
                test_acc = 0.0
                valid_acc = 0.0

            # Build network and apply bulletproof wrapper
            # Note: When building the network for cifar10, 'cifar10-valid' or 'cifar10' must be used to pull the config
            config_key = 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset
            config = nas201_api.get_net_config(arch_index, config_key)
            raw_model = get_cell_based_tiny_net(config)
            model = NAS201Wrapper(raw_model).cuda(args.gpu)

            # ==========================================
            # 🎯 Algorithm distribution and invocation section
            # ==========================================
            if args.proxy == 'SCS':
                score = compute_scs_score.compute_scs(model, input_image_size, batch_size, mixup_gamma=1e-2,
                                                      gpu=args.gpu)
            elif args.proxy == 'ISR':
                score = compute_isr_score.compute_isr(model, input_image_size, batch_size, mixup_gamma=1e-2, repeat=1,
                                                      gpu=args.gpu)
            elif args.proxy == 'ZiCo':
                score = compute_zico_score.compute_nas_score(gpu=args.gpu, model=model, resolution=input_image_size,
                                                             batch_size=batch_size)
            elif args.proxy == 'Zen':
                score = compute_zen_score.compute_zen_score(gpu=args.gpu, model=model, resolution=input_image_size,
                                                            batch_size=batch_size, mixup_gamma=1e-2, repeat=1)
            elif args.proxy == 'TE-NAS':
                score = compute_te_nas_score.compute_NTK_score(gpu=args.gpu, model=model, resolution=input_image_size,
                                                               batch_size=batch_size)
            elif args.proxy == 'Syncflow':
                score = compute_syncflow_score.do_compute_nas_score(gpu=args.gpu, model=model,
                                                                    resolution=input_image_size, batch_size=batch_size)
            elif args.proxy == 'GradNorm':
                score = compute_gradnorm_score.compute_nas_score(gpu=args.gpu, model=model, resolution=input_image_size,
                                                                 batch_size=batch_size)
            elif args.proxy == 'NASWOT':
                score = compute_NASWOT_score.compute_nas_score(gpu=args.gpu, model=model, resolution=input_image_size,
                                                               batch_size=batch_size)
            else:
                score = 0.0

            # Display Valid accuracy during printing
            print(f"  📊 Valid Acc: {valid_acc:5.2f}% | Test Acc: {test_acc:5.2f}% | {args.proxy} Score: {score:10.4f}")

            results_raw.append({
                'name': name,
                'arch': arch,
                'valid_acc': valid_acc,
                'test_acc': test_acc,
                'score': score
            })

            del raw_model, model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Sort in descending order based on the calculated scores
    sorted_results = sorted(results_raw, key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 160)
    print(f"🏆 {args.proxy} Batch Validation Leaderboard (Dataset: {args.dataset})")
    print("=" * 160)

    for rank, res in enumerate(sorted_results, 1):
        is_high_acc = "🔥" if (args.dataset == 'ImageNet16-120' and res['test_acc'] >= 45.0) or \
                              (args.dataset == 'cifar10' and res['test_acc'] >= 94.0) or \
                              (args.dataset == 'cifar100' and res['test_acc'] >= 72.0) else "  "

        # 🌟 Key point: Format and output both valid_acc and test_acc here
        print(
            f"Top {rank:3d} {is_high_acc:3s} | "
            f"{args.proxy} Score: {res['score']:12.4f} | "
            f"Valid Acc: {res['valid_acc']:5.2f}% | "
            f"Test Acc: {res['test_acc']:5.2f}% | "
            f"Arch Name: {res['name']:<25} | "
            f"Arch: {res['arch']}"
        )


if __name__ == '__main__':
    test_batch_proxies()