import os
import sys
import argparse
import random
import re
import math
from pathlib import Path

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn

from reproducibility import (
    DEFAULT_PROXY_BASE_SEED,
    proxy_rng_seed,
    search_rng_seed,
    seed_everything,
    write_json,
)
class NAS201Wrapper(nn.Module):
    """Expose NAS-Bench-201 classification logits to zero-cost proxies."""

    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = int(num_classes)

    def forward(self, x):
        out = self.model(x)

        # xautodl's NAS-Bench-201 TinyNetwork returns (features, logits).
        # The zero-cost proxy must operate on the classification logits.
        if isinstance(out, tuple):
            if len(out) < 2:
                raise RuntimeError("Unexpected NAS-Bench-201 model output tuple.")
            logits = out[1]
        else:
            logits = out

        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise RuntimeError(
                f"Unexpected classifier output shape {tuple(logits.shape)}; "
                f"expected [batch, {self.num_classes}]."
            )
        return logits


def get_random_nas201_arch(rng):
    OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return f"|{rng.choice(OPS)}~0|+|{rng.choice(OPS)}~0|{rng.choice(OPS)}~1|+|{rng.choice(OPS)}~0|{rng.choice(OPS)}~1|{rng.choice(OPS)}~2|"


def get_unseen_random_nas201_arch(rng, visited):
    for _ in range(10000):
        arch = get_random_nas201_arch(rng)
        if arch not in visited:
            return arch
    raise RuntimeError("Unable to sample an unseen NAS-Bench-201 architecture.")
def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120'])
    parser.add_argument('--api_path', type=str, default='./NAS-Bench-201-v1_1-096897.pth')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--proxy_base_seed', type=int, default=DEFAULT_PROXY_BASE_SEED)
    parser.add_argument('--evolution_max_iter', type=int, default=300)
    parser.add_argument('--initial_random', type=int, default=51)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--patience_mode', type=str, default='heuristic', choices=['heuristic', 'fixed'])
    parser.add_argument('--patience_start', type=int, default=200,
                        help='Warm-up iteration used only when --patience_mode=fixed.')
    parser.add_argument('--patience', type=int, default=50,
                        help='Fixed non-improvement limit used only when --patience_mode=fixed.')
    parser.add_argument('--phase1_prompt_path', type=str, default='./prompt/prompt.txt')
    parser.add_argument('--phase2_prompt_path', type=str, default='./prompt_tribunal.txt')
    parser.add_argument('--phase1_model', type=str, default='deepseek-chat')
    parser.add_argument('--phase2_model', type=str, default='deepseek-reasoner')
    parser.add_argument('--phase1_temperature', type=float, default=1.0)
    parser.add_argument('--phase2_temperature', type=float, default=0.0)
    parser.add_argument('--phase1_max_tokens', type=int, default=500)
    parser.add_argument('--phase2_max_tokens', type=int, default=1500)
    parser.add_argument('--llm_retries', type=int, default=3)
    parser.add_argument('--llm_timeout', type=float, default=120.0)
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args(argv)

    if args.dataset in {'cifar10', 'cifar10-valid'}:
        args.num_classes, args.input_image_size = 10, 32
        args.dataset_key, args.api_dataset = 'cifar10', 'cifar10-valid'
    elif args.dataset == 'cifar100':
        args.num_classes, args.input_image_size = 100, 32
        args.dataset_key = args.api_dataset = 'cifar100'
    elif args.dataset == 'ImageNet16-120':
        args.num_classes, args.input_image_size = 120, 16
        args.dataset_key = args.api_dataset = 'ImageNet16-120'

    if args.initial_random <= 0:
        parser.error('--initial_random must be positive')
    if args.population_size < args.initial_random:
        parser.error('--population_size must be at least --initial_random')
    if args.top_k <= 0 or args.top_k > args.population_size:
        parser.error('--top_k must be in [1, population_size]')

    return args
def compute_proxy_score(the_model, gpu, proxy_name, args):
    from ZeroShotProxy import compute_zico_score

    the_model = the_model.cuda(gpu)
    try:
        if proxy_name == 'ZiCo':
            score = compute_zico_score.compute_nas_score(gpu=gpu, model=the_model, resolution=args.input_image_size,
                                                         batch_size=args.batch_size)
        else:
            score = 1e-4
        if np.isnan(score) or np.isinf(score): score = 1e-4
    except Exception as exc:
        print(f"[Proxy warning] ZiCo evaluation failed: {exc}")
        score = 1e-4

    del the_model
    torch.cuda.empty_cache()
    return score


# LLM Invocation Module (Evolutionary Mutation + Final Tribunal)
def generate_by_llm(args, structure_str, score, num_replaces, failed_attempts=None, local_references=None):
    if failed_attempts is None: failed_attempts = []
    if local_references is None: local_references = []
    from openai import OpenAI
    with open(args.phase1_prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{architecture}}", structure_str)
    prompt = prompt.replace("{{score}}", str(score))
    prompt = prompt.replace('{{mutate_num}}', str(num_replaces))

    # Keep Phase-1 context bounded: three refreshed population references at most.
    if local_references:
        prompt += "\n\n=== LOCAL POPULATION REFERENCES ===\n"
        for ref_id, (ref_arch, ref_score) in enumerate(local_references):
            prompt += f"Reference {ref_id}: score={ref_score}; architecture={ref_arch}\n"

    if len(failed_attempts) > 0:
        prompt += "\n\n=== SYSTEM FEEDBACK ===\nDO NOT GENERATE THESE REJECTED PROPOSALS AGAIN:\n" + "\n".join(
            failed_attempts)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=args.llm_timeout)
    response = client.chat.completions.create(
        model=args.phase1_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.phase1_temperature,
        max_tokens=args.phase1_max_tokens,
    )
    return response.choices[0].message.content


def semantic_tribunal_by_llm(args, candidates_json_str, dataset_name):
    from openai import OpenAI
    with open(args.phase2_prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{dataset_name}}", dataset_name)
    prompt = prompt.replace("{{candidates_data}}", candidates_json_str)

    print(f"\n[LLM Semantic Tribunal] Reviewing {args.top_k} elite topologies for {dataset_name}...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=args.llm_timeout)
    response = client.chat.completions.create(
        model=args.phase2_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.phase2_temperature,
        max_tokens=args.phase2_max_tokens,
    )
    return response.choices[0].message.content
# Core main loop: Phase 1 proxy-guided evolution, then Phase 2 reranking.
def main(args):
    from nas_201_api import NASBench201API as API
    from xautodl.models import get_cell_based_tiny_net

    gpu = args.gpu
    torch.cuda.set_device(f'cuda:{gpu}')

    api_path = Path(args.api_path).resolve()
    if not api_path.is_file():
        raise FileNotFoundError(f"NAS-Bench-201 API file not found: {api_path}")
    nas201_api = API(str(api_path))

    run_rng_seed = search_rng_seed(args.dataset_key, args.seed)
    rng = random.Random(run_rng_seed)
    phase2_shuffle_seed = run_rng_seed ^ 0x5EED201
    seed_everything(run_rng_seed)

    # The manuscript's adaptive patience heuristic is supported directly.
    # Fixed patience remains available for controlled/legacy runs.
    if args.patience_mode == 'heuristic':
        patience_limit = math.floor(35 + math.log10(15625) + 0.1 * args.num_classes)
    else:
        patience_limit = args.patience

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / 'run_config.json',
        {
            'dataset': args.dataset_key,
            'api_dataset': args.api_dataset,
            'reported_seed': args.seed,
            'derived_search_rng_seed': run_rng_seed,
            'proxy_base_seed': args.proxy_base_seed,
            'evolution_max_iter': args.evolution_max_iter,
            'initial_random': args.initial_random,
            'population_size': args.population_size,
            'batch_size': args.batch_size,
            'top_k': args.top_k,
            'patience_mode': args.patience_mode,
            'patience_start': args.patience_start if args.patience_mode == 'fixed' else None,
            'patience_limit': patience_limit,
            'phase2_shuffle_seed': phase2_shuffle_seed,
            'phase1_model': args.phase1_model,
            'phase1_temperature': args.phase1_temperature,
            'phase2_model': args.phase2_model,
            'phase2_temperature': args.phase2_temperature,
            'candidate_order': 'deterministically shuffled after proxy Top-K selection',
            'phase2_withheld_fields': [
                'numerical proxy score',
                'validation accuracy',
                'test accuracy',
            ],
        },
    )

    popu_structure_list = []
    popu_zico_score_list = []
    global_arch_history = set()

    print(f"\nAgentic NAS initialized. Target dataset: {args.dataset}")
    print("Entering Phase 1: ZiCo gradient-driven evolution")

    patience_counter = 0
    best_phase_score = float('-inf')

    for loop_count in range(args.evolution_max_iter):

        while len(popu_structure_list) > args.population_size:
            tmp_idx = popu_zico_score_list.index(min(popu_zico_score_list))
            popu_zico_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)

        current_pool_size = len(popu_structure_list)

        if current_pool_size < args.initial_random:
            random_structure_str = get_unseen_random_nas201_arch(rng, global_arch_history)
            is_valid_new_arch = True
        else:
            tmp_idx = rng.randint(0, current_pool_size - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            tmp_score = popu_zico_score_list[tmp_idx]

            # Three refreshed local population references, excluding the parent.
            candidate_ref_indices = [i for i in range(current_pool_size) if i != tmp_idx]
            ref_count = min(3, len(candidate_ref_indices))
            ref_indices = rng.sample(candidate_ref_indices, ref_count) if ref_count else []
            local_references = [
                (popu_structure_list[i], popu_zico_score_list[i])
                for i in ref_indices
            ]

            retry_count = 0
            is_valid_new_arch = False
            current_failed_attempts = []
            current_num_replaces = 1

            while retry_count < args.llm_retries:
                llm_raw_output = generate_by_llm(
                    args,
                    tmp_random_structure_str,
                    tmp_score,
                    current_num_replaces,
                    failed_attempts=current_failed_attempts,
                    local_references=local_references,
                )
                match = re.search(r'"(?:arch|architecture)"\s*:\s*"([^"]+)"', llm_raw_output)
                if match:
                    parsed_arch = match.group(1).replace(" ", "").replace("\n", "").replace("\\", "")

                    # Reject malformed/out-of-space proposals before proxy evaluation.
                    try:
                        parsed_idx = nas201_api.query_index_by_arch(parsed_arch)
                        is_search_space_valid = parsed_idx is not None and parsed_idx >= 0
                    except Exception:
                        is_search_space_valid = False

                    if is_search_space_valid and parsed_arch not in global_arch_history:
                        random_structure_str = parsed_arch
                        is_valid_new_arch = True
                        break

                    current_failed_attempts.append(parsed_arch)
                    current_num_replaces = min(3, current_num_replaces + 1)
                retry_count += 1

            if not is_valid_new_arch:
                random_structure_str = get_unseen_random_nas201_arch(rng, global_arch_history)
        global_arch_history.add(random_structure_str)

        try:
            score_seed = proxy_rng_seed(args.proxy_base_seed, args.dataset_key, random_structure_str)
            seed_everything(score_seed)
            cfg = nas201_api.get_net_config(
                nas201_api.query_index_by_arch(random_structure_str),
                args.api_dataset,
            )
            the_model = NAS201Wrapper(get_cell_based_tiny_net(cfg), args.num_classes)
            the_zico_score = compute_proxy_score(the_model, gpu, 'ZiCo', args)
        except Exception as exc:
            print(f"[Search warning] Failed to score architecture {random_structure_str}: {exc}")
            the_zico_score = 1e-4

        popu_structure_list.append(random_structure_str)
        popu_zico_score_list.append(the_zico_score)

        # Initialize the stagnation reference from the completed random population.
        if len(popu_structure_list) == args.initial_random and best_phase_score == float('-inf'):
            best_phase_score = max(popu_zico_score_list)
            patience_counter = 0

        # Stagnation monitoring for mutation/evolution steps.
        elif current_pool_size >= args.initial_random:
            if the_zico_score > best_phase_score:
                best_phase_score = the_zico_score
                patience_counter = 0
            else:
                should_count = (
                    args.patience_mode == 'heuristic'
                    or loop_count >= args.patience_start
                )
                if should_count:
                    patience_counter += 1

        print(
            f"Iter: {loop_count:3d} | Proxy: ZiCo | Score: {the_zico_score:8.4f} "
            f"| Patience: {patience_counter}/{patience_limit}"
        )

        should_stop = (
            patience_counter >= patience_limit
            and (
                args.patience_mode == 'heuristic'
                or loop_count >= args.patience_start
            )
        )
        if should_stop:
            print("\n[Evolution Terminated] Early Stop triggered. ZiCo reached a bottleneck.")
            break

    # Phase 2: identifier/genotype-only semantic reranking via LLM.
    print("\nEntering Phase 2: semantic topology-aware reranking")

    # Use the proxy only to form the Top-K set. Before the Tribunal call,
    # deterministically shuffle and re-index candidates so neither score nor
    # proxy-derived rank/order is exposed to the LLM.
    combined_pool = list(zip(popu_structure_list, popu_zico_score_list))
    combined_pool.sort(key=lambda x: x[1], reverse=True)
    ranked_top_k = [
        {'proxy_rank': rank, 'genotype': arch}
        for rank, (arch, _score) in enumerate(combined_pool[:args.top_k])
    ]
    phase2_rng = random.Random(phase2_shuffle_seed)
    phase2_rng.shuffle(ranked_top_k)
    top_k_archs = [item['genotype'] for item in ranked_top_k]

    # Generate the identifier/genotype-only Tribunal input.
    candidates_str = ""
    for i, arch in enumerate(top_k_archs):
        candidates_str += f"{i}. {arch}\n"

    write_json(
        output_dir / 'phase2_candidates.json',
        {
            'dataset': args.dataset_key,
            'reported_seed': args.seed,
            'candidate_order': 'deterministically shuffled after proxy Top-K selection',
            'phase2_shuffle_seed': phase2_shuffle_seed,
            'scores_in_llm_input': False,
            'accuracies_in_llm_input': False,
            'proxy_rank_in_llm_input': False,
            'candidates': [
                {
                    'candidate_id': i,
                    'genotype': item['genotype'],
                    # Retained only in the local audit log; not included in candidates_str.
                    'posthoc_proxy_rank': item['proxy_rank'],
                }
                for i, item in enumerate(ranked_top_k)
            ],
        },
    )
    print(f"Submitting {args.top_k} identifier/genotype candidates to DeepSeek...")
    # Call the LLM judge.
    llm_judgment_raw = semantic_tribunal_by_llm(args, candidates_str, args.dataset_key)

    # Parse the selection. A malformed response aborts the run instead of
    # silently defaulting to the highest-proxy candidate (ID 0).
    match = re.search(r'"winner_id"\s*:\s*"?(\d+)"?', llm_judgment_raw)
    if not match:
        raise RuntimeError("LLM output does not contain a valid winner_id.")
    best_id = int(match.group(1))
    if not 0 <= best_id < len(top_k_archs):
        raise RuntimeError(f"winner_id {best_id} is outside the candidate range.")

    best_arch = top_k_archs[best_id]

    write_json(
        output_dir / 'phase2_selection.json',
        {
            'dataset': args.dataset_key,
            'reported_seed': args.seed,
            'winner_id': best_id,
            'winner_genotype': best_arch,
            'raw_response': llm_judgment_raw,
        },
    )

    print("\nTribunal Judgment:")
    print(llm_judgment_raw)

    # Final Stage: True Accuracy Reveal
    print(f"\nPost-hoc accuracy lookup for {args.top_k} elite architectures ({args.dataset_key})")

    for i, arch in enumerate(top_k_archs):
        arch_index = nas201_api.query_index_by_arch(arch)
        final_info = nas201_api.get_more_info(
            arch_index,
            args.api_dataset,
            hp='200',
            is_random=False,
        )

        # Extract Test Accuracy and Valid Accuracy respectively
        test_acc = final_info.get('test-accuracy', final_info.get('valtest-accuracy', 0.0))
        valid_acc = final_info.get('valid-accuracy', 0.0)

        # Mark the architecture selected by the LLM
        mark = "[SELECTED]" if i == best_id else "  "

        print(f"{mark:<10} ID: {i} | Valid: {valid_acc:>5.2f}% | Test: {test_acc:>5.2f}% | Arch: {arch}")

    print("\nAgentic LLM Tribunal NAS successfully completed.")


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv[1:])
    main(args)
