import os
import sys
import argparse, time
import torch
import numpy as np
import secrets
import logging
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import global_utils
import torch.nn as nn

rng = secrets.SystemRandom()

# NAS-Bench-201 API and model acquisition
from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net
from ZeroShotProxy import compute_zico_score


class NAS201Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


nas201_api = API('NAS-Bench-201-v1_1-096897.pth')


def get_random_nas201_arch():
    OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return f"|{secrets.choice(OPS)}~0|+|{secrets.choice(OPS)}~0|{secrets.choice(OPS)}~1|+|{secrets.choice(OPS)}~0|{secrets.choice(OPS)}~1|{secrets.choice(OPS)}~2|"


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'])
    parser.add_argument('--evolution_max_iter', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./output')
    args, _ = parser.parse_known_args(argv)

    if args.dataset == 'cifar10-valid':
        args.num_classes, args.input_image_size = 10, 32
    elif args.dataset == 'cifar100':
        args.num_classes, args.input_image_size = 100, 32
    elif args.dataset == 'ImageNet16-120':
        args.num_classes, args.input_image_size = 120, 16

    return args


def compute_proxy_score(the_model, gpu, proxy_name, args):
    the_model = the_model.cuda(gpu)
    try:
        if proxy_name == 'ZiCo':
            score = compute_zico_score.compute_nas_score(gpu=gpu, model=the_model, resolution=args.input_image_size,
                                                         batch_size=args.batch_size)
        else:
            score = 1e-4
        if np.isnan(score) or np.isinf(score): score = 1e-4
    except Exception:
        score = 1e-4

    del the_model
    torch.cuda.empty_cache()
    return score


# LLM Invocation Module (Evolutionary Mutation + Final Tribunal)
def generate_by_llm(structure_str, score, num_replaces, failed_attempts=None):
    if failed_attempts is None: failed_attempts = []
    file_path = "/data/XuZiJie/rznas/prompt/prompt.txt"
    from openai import OpenAI
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{architecture}}", structure_str)
    prompt = prompt.replace("{{score}}", str(score))
    prompt = prompt.replace('{{mutate_num}}', str(num_replaces))

    if len(failed_attempts) > 0:
        prompt += "\n\n=== SYSTEM FEEDBACK ===\nDO NOT GENERATE THESE EXACT DUPLICATES AGAIN:\n" + "\n".join(
            failed_attempts)

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com")
    response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
                                              temperature=0.6, max_tokens=500)
    return response.choices[0].message.content


def semantic_tribunal_by_llm(candidates_json_str, dataset_name):
    file_path = "prompt_tribunal.txt"
    from openai import OpenAI
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{dataset_name}}", dataset_name)
    prompt = prompt.replace("{{candidates_data}}", candidates_json_str)

    print(f"\n[LLM Semantic Tribunal] Reviewing top 10 elite topologies for {dataset_name}...")
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com")
    response = client.chat.completions.create(model="", messages=[{"role": "user", "content": prompt}],
                                              temperature=0.2, max_tokens=600)
    return response.choices[0].message.content

# Core Main Loop (Phase 1: ZiCo Evolution -> Phase 2: LLM Pure Topological Tribunal)
def main(args):
    gpu = args.gpu
    torch.cuda.set_device(f'cuda:{gpu}')

    popu_structure_list = []
    popu_zico_score_list = []
    global_arch_history = set()

    print(f"\nAgentic NAS initialized. Target dataset: {args.dataset}")
    print("Entering Phase 1: ZiCo gradient-driven evolution")

    patience_counter = 0
    best_phase_score = -999999.0

    for loop_count in range(args.evolution_max_iter):

        while len(popu_structure_list) > args.population_size:
            tmp_idx = popu_zico_score_list.index(min(popu_zico_score_list))
            popu_zico_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)

        current_pool_size = len(popu_structure_list)

        if current_pool_size <= 50:
            random_structure_str = get_random_nas201_arch()
            is_valid_new_arch = True
        else:
            tmp_idx = rng.randint(0, current_pool_size - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            tmp_score = popu_zico_score_list[tmp_idx]

            retry_count = 0
            is_valid_new_arch = False
            current_failed_attempts = []
            current_num_replaces = 1

            while retry_count < 3:
                llm_raw_output = generate_by_llm(tmp_random_structure_str, tmp_score, current_num_replaces,
                                                 failed_attempts=current_failed_attempts)
                import re
                match = re.search(r'"arch"\s*:\s*"([^"]+)"', llm_raw_output)
                if match:
                    parsed_arch = match.group(1).replace(" ", "").replace("\n", "").replace("\\", "")
                    if parsed_arch not in global_arch_history:
                        random_structure_str = parsed_arch
                        is_valid_new_arch = True
                        break
                    else:
                        current_failed_attempts.append(parsed_arch)
                        current_num_replaces = min(3, current_num_replaces + 1)
                retry_count += 1

            if not is_valid_new_arch:
                random_structure_str = get_random_nas201_arch()

        global_arch_history.add(random_structure_str)

        try:
            cfg = nas201_api.get_net_config(nas201_api.query_index_by_arch(random_structure_str), args.dataset)
            the_model = NAS201Wrapper(get_cell_based_tiny_net(cfg))
            the_zico_score = compute_proxy_score(the_model, gpu, 'ZiCo', args)
        except Exception:
            the_zico_score = 1e-4

        popu_structure_list.append(random_structure_str)
        popu_zico_score_list.append(the_zico_score)

        # Stagnation monitoring mechanism
        if current_pool_size > 50:
            if the_zico_score > best_phase_score:
                best_phase_score = the_zico_score
                patience_counter = 0
            else:
                # Accumulate patience only after 200 iterations
                if loop_count >= 200:
                    patience_counter += 1

        print(f"Iter: {loop_count:3d} | Proxy: ZiCo | Score: {the_zico_score:8.4f} | Patience: {patience_counter}/50")

        if patience_counter >= 50 and loop_count >= 200:
            print("\n[Evolution Terminated] Early Stop triggered. ZiCo reached a bottleneck.")
            break

    # Phase 2: Pure Topological Double-Blind Test via LLM
    print("\nEntering Phase 2: LLM Pure Topological Intuition Test")

    # 1. Extract Top 10 from Phase 1 (Keep only architectures, discard scores)
    combined_pool = list(zip(popu_structure_list, popu_zico_score_list))
    combined_pool.sort(key=lambda x: x[1], reverse=True)
    top_10_archs = [x[0] for x in combined_pool[:10]]

    # 2. Generate pure architecture list string
    candidates_str = ""
    for i, arch in enumerate(top_10_archs):
        candidates_str += f"{i}. {arch}\n"

    print("Submitting Top 10 pure topologies to DeepSeek for physical reasoning...")

    # 3. Call LLM Judge
    llm_judgment_raw = semantic_tribunal_by_llm(candidates_str, args.dataset)

    # 4. Parse results
    import re
    best_id = 0
    try:
        match = re.search(r'"winner_id"\s*:\s*(\d+)', llm_judgment_raw)
        if match:
            best_id = int(match.group(1))
    except:
        print("LLM output parsing failed, defaulting to ID 0.")

    best_arch = top_10_archs[best_id]

    print("\nTribunal Judgment:")
    print(llm_judgment_raw)

    # Final Stage: True Accuracy Reveal
    print(f"\nRevealing True Accuracy for Top 10 Elite Architectures ({args.dataset})")

    for i, arch in enumerate(top_10_archs):
        arch_index = nas201_api.query_index_by_arch(arch)
        final_info = nas201_api.get_more_info(arch_index, args.dataset, hp='200', is_random=False)

        # Extract Test Accuracy and Valid Accuracy respectively
        test_acc = final_info.get('test-accuracy', final_info.get('valtest-accuracy', 0.0))
        valid_acc = final_info.get('valid-accuracy', 0.0)

        # Mark the architecture selected by the LLM
        mark = "[LLM SOTA]" if i == best_id else "  "

        print(f"{mark:<10} ID: {i} | Valid: {valid_acc:>5.2f}% | Test: {test_acc:>5.2f}% | Arch: {arch}")

    print("\nAgentic LLM Tribunal NAS successfully completed.")


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    main(args)