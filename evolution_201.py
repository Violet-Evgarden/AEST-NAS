import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
import numpy as np
import global_utils
# Keep PlainNet import in case ZeroShotProxy has implicit dependencies
import Masternet
import torch.nn as nn
from ZeroShotProxy import compute_opse_score


# Model wrapper designed for NAS-Bench-201
class NAS201Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # If the model returns a tuple (logits, features), force extraction of the first logits tensor
        if isinstance(out, tuple):
            return out[0]
        return out


from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net

# Global database loading
print("Loading NAS-Bench-201 database, please wait...")
nas201_api = API('NAS-Bench-201-v1_1-096897.pth')
print("Loading complete.")

from ZeroShotProxy import compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, \
    compute_NASWOT_score, compute_zico_score, compute_zen_score, compute_isr_score, compute_scs_score

working_dir = os.path.dirname(os.path.abspath(__file__))

import secrets


def get_random_nas201_arch():
    """Generate a completely random architecture using the system's physical entropy source."""
    OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    op0 = secrets.choice(OPS)
    op1 = secrets.choice(OPS)
    op2 = secrets.choice(OPS)
    op3 = secrets.choice(OPS)
    op4 = secrets.choice(OPS)
    op5 = secrets.choice(OPS)

    return f"|{op0}~0|+|{op1}~0|{op2}~1|+|{op3}~0|{op4}~1|{op5}~2|"


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--zero_shot_score', type=str, default='SCS',
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default='SearchSpace/search_space_IDW_fixfc.py',
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=150,
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=32,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=100, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')

    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def generate_by_llm(structure_str, score, num_replaces, dynamic_samples=None, failed_attempts=None):
    if dynamic_samples is None:
        dynamic_samples = []
    if failed_attempts is None: failed_attempts = []

    file_path = " "
    import os
    from openai import OpenAI

    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{architecture}}", structure_str)
    prompt = prompt.replace("{{score}}", str(score))
    prompt = prompt.replace('{{mutate_num}}', str(num_replaces))

    memory_injection = "\n\n=== HISTORICAL SAMPLES FOR REFERENCE ==="
    memory_injection += "\nStudy these recent samples to understand the relationship between architecture patterns and scores:\n\n"

    if len(dynamic_samples) > 0:
        for i, record in enumerate(dynamic_samples):
            memory_injection += f"Sample {i + 1}: Score={record['score']:.4f} | Arch={record['arch']}\n"
    else:
        memory_injection += "No historical data available yet. Start exploring now!\n"
    prompt += memory_injection

    # Error tracking and hard constraint injection
    if len(failed_attempts) > 0:
        error_feedback = "\n\n=== SYSTEM FEEDBACK: DUPLICATE ARCHITECTURE DETECTED ===\n"
        error_feedback += "The following architectures you just generated are exact duplicates of our existing population.\n"
        error_feedback += "To effectively explore the search space, you MUST NOT generate these architectures again:\n\n"

        for i, bad_arch in enumerate(failed_attempts):
            error_feedback += f"[DUPLICATE {i + 1}]: {bad_arch}\n"

        error_feedback += "\nConstraint: You must generate a COMPLETELY NOVEL architecture string. Do not repeat the duplicates listed above.\n"
        prompt += error_feedback

    print("Final Prompt Sent to LLM:\n")
    print(prompt)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=500,
        top_p=0.9,
        stop=["<|im_end|>"]
    )
    new_structure_str = response.choices[0].message.content
    return new_structure_str


def get_new_random_structure_str(AnyPlainNet, structure_str, score, num_classes, get_search_space_func,
                                 num_replaces, dynamic_samples=None, failed_attempts=None):
    get_new_llm_structure_str = generate_by_llm(structure_str, score, num_replaces, dynamic_samples=dynamic_samples,
                                                failed_attempts=failed_attempts)
    return get_new_llm_structure_str


def compute_nas_score(the_model, gpu, args):
    the_model = the_model.cuda(gpu)
    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core = compute_zen_score.compute_zen_score(model=the_model, gpu=gpu,
                                                               resolution=args.input_image_size,
                                                               mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                               repeat=1)
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
        elif args.zero_shot_score == 'Syncflow':
            the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                       resolution=args.input_image_size,
                                                                       batch_size=args.batch_size)
        elif args.zero_shot_score == 'GradNorm':
            the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)
        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
        elif args.zero_shot_score == 'ZiCo':
            the_nas_core = compute_zico_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                resolution=args.input_image_size,
                                                                batch_size=args.batch_size)
        elif args.zero_shot_score == 'ISR':
            # ISR internal structural resonance scoring (adapted for search loops)
            the_nas_core = compute_isr_score.compute_isr(
                model=the_model,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
                mixup_gamma=args.gamma,
                repeat=32,
                gpu=gpu
            )
        elif args.zero_shot_score == 'OPSE':
            the_nas_core = compute_opse_score.compute_opse(
                model=the_model,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
                mixup_gamma=args.gamma,
                gpu=gpu
            )
        elif args.zero_shot_score == 'SCS':
            the_nas_core = compute_scs_score.compute_scs(
                model=the_model,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
                mixup_gamma=args.gamma,
                gpu=gpu
            )
        else:
            the_nas_core = np.random.randn()

    except Exception as err:
        logging.info(str(err))
        the_nas_core = -9999

    del the_model
    torch.cuda.empty_cache()
    return the_nas_core


def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    select_search_space = global_utils.load_py_module_from_path(args.search_space)
    AnyPlainNet = Masternet.MasterNet

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()

    # Establish a global gene pool outside the main loop
    global_arch_history = set()

    for loop_count in range(args.evolution_max_iter):
        # Dynamic annealing mutation step (Simulated Annealing)
        cold_start_size = 50
        if loop_count < cold_start_size:
            progress_ratio = 0.0
        else:
            progress_ratio = (loop_count - cold_start_size) / (args.evolution_max_iter - cold_start_size)

        if progress_ratio < 0.4:
            num_replaces = random.choice([2, 3])
        elif progress_ratio < 0.8:
            num_replaces = random.choice([1, 2])
        else:
            num_replaces = random.choices([1, 2], weights=[0.8, 0.2])[0]

        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            if popu_latency_list:
                popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(
                f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time / 3600:4g}h')

        current_pool_size = len(popu_structure_list)

        if current_pool_size > 0:
            sample_num = min(6, current_pool_size)
            selected_indices = random.sample(range(current_pool_size), sample_num)

            dynamic_samples = []
            for idx in selected_indices:
                dynamic_samples.append({
                    "arch": popu_structure_list[idx],
                    "score": popu_zero_shot_score_list[idx]
                })
        else:
            dynamic_samples = []

        if len(popu_structure_list) <= 50:
            print(f"\n[Cold Start Phase] Generating random initial architecture {len(popu_structure_list) + 1}/51...")
            random_structure_str = get_random_nas201_arch()
            print("Extracted network structure:", random_structure_str)

        else:
            pool_size = len(popu_structure_list)

            if progress_ratio < 0.5:
                random_explore_prob = 0.90
            elif progress_ratio < 0.95:
                random_explore_prob = 0.80
            else:
                random_explore_prob = 0.60

            magic_dice = random.random()
            if magic_dice < random_explore_prob:
                tmp_idx = random.randint(0, pool_size - 1)
                print(
                    f"\n[Iteration {loop_count} | Exploration Rate {random_explore_prob * 100}%] Random selection for mutation (Selected index {tmp_idx}, Score: {popu_zero_shot_score_list[tmp_idx]:.4f})...")
            else:
                candidates = random.sample(range(pool_size), min(3, pool_size))
                tmp_idx = max(candidates, key=lambda idx: popu_zero_shot_score_list[idx])
                print(
                    f"\n[Iteration {loop_count} | Exploration Rate {random_explore_prob * 100}%] Elite selection for mutation (Score: {popu_zero_shot_score_list[tmp_idx]:.4f})...")

            tmp_random_structure_str = popu_structure_list[tmp_idx]
            tmp_score = popu_zero_shot_score_list[tmp_idx]

            # Retry mechanism with In-Context Feedback for duplication checking
            max_retries = 3
            retry_count = 0
            is_valid_new_arch = False
            current_failed_attempts = []

            while retry_count < max_retries:
                llm_raw_output = get_new_random_structure_str(
                    AnyPlainNet=AnyPlainNet, structure_str=tmp_random_structure_str, score=tmp_score,
                    num_classes=args.num_classes,
                    get_search_space_func=select_search_space.gen_search_space, num_replaces=num_replaces,
                    dynamic_samples=dynamic_samples,
                    failed_attempts=current_failed_attempts
                )

                print("\nDeepSeek Raw Output:")
                print(llm_raw_output)
                print("\n")

                import re
                parsed_arch = None
                try:
                    match = re.search(r'"arch"\s*:\s*"([^"]+)"', llm_raw_output)
                    if match:
                        parsed_arch = match.group(1).replace(" ", "").replace("\n", "").replace("\\", "")
                except Exception as e:
                    print("Parsing error:", e)

                if not parsed_arch:
                    print("Parsing failed: 'arch' field not found. Retrying...")
                    retry_count += 1
                    continue

                print(f"[Instruction Monitoring] Requested modifications: {num_replaces}")
                print(f"[Parent Architecture] Input: {tmp_random_structure_str}")
                print(f"[Child Architecture] Output: {parsed_arch}")

                if parsed_arch in global_arch_history:
                    retry_count += 1
                    print(
                        f"[Duplication Intercepted] LLM generated a repeated architecture. ({retry_count}/{max_retries})")
                    current_failed_attempts.append(parsed_arch)

                    if num_replaces < 3:
                        num_replaces += 1
                        print(
                            f"[Dynamic Adjustment] Forcing mutation step increase to {num_replaces} to escape local optima.")
                else:
                    random_structure_str = parsed_arch
                    is_valid_new_arch = True
                    break

            if not is_valid_new_arch:
                print("[Retry Failed] LLM failed 3 consecutive times. Falling back to random physical generation.")
                random_structure_str = get_random_nas201_arch()

        global_arch_history.add(random_structure_str)

        the_model = None
        try:
            arch_index = nas201_api.query_index_by_arch(random_structure_str)
            config = nas201_api.get_net_config(arch_index, 'cifar10')
            the_model = get_cell_based_tiny_net(config)

            # Prevent tuple errors during calculation
            the_model = NAS201Wrapper(the_model)

        except Exception as e:
            print(f"[Warning] Invalid or non-existent LLM generated architecture skipped: {random_structure_str}")
            continue

        try:
            the_latency = 0
            the_nas_core = compute_nas_score(the_model, gpu, args)
        except Exception as e:
            print(f"[Scoring Error] {e}")
            the_nas_core = -9999

        # Record search results to log file
        logging.info(f"Iter: {loop_count:4d} | Score: {the_nas_core:8.4f} | Arch: {random_structure_str}")
        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # Evolutionary search complete. Entering Phase 2: Top 10 Two-Stage Verification
    print("\nEvolutionary search complete! Initiating Phase 2: 12-Epoch Two-Stage Verification for Top 10.")

    history_archs = []
    for arch, score in zip(popu_structure_list, popu_zero_shot_score_list):
        history_archs.append({'arch': arch, 'score': score})

    top_10_candidates = sorted(history_archs, key=lambda x: x['score'], reverse=True)[:10]

    verified_results = []

    print("\n[Phase 1] SCS Proxy Scoring Results:")
    for rank, item in enumerate(top_10_candidates, 1):
        arch = item['arch']
        scs_score = item['score']
        arch_index = nas201_api.query_index_by_arch(arch)

        try:
            # Query validation accuracy at Epoch 12 (iepoch=11)
            epoch_12_info = nas201_api.get_more_info(arch_index, 'cifar10-valid', iepoch=11, hp='200', is_random=False)
            valid_acc_12e = epoch_12_info['valid-accuracy']

            final_test_acc = nas201_api.get_more_info(arch_index, 'cifar10', hp='200', is_random=False)['test-accuracy']

            print(
                f"  SCS Rank {rank:2d} | Proxy Score: {scs_score:8.0f} | 12-Epoch Valid Acc: {valid_acc_12e:5.2f}% | Arch: {arch}")

            verified_results.append({
                'scs_rank': rank,
                'arch': arch,
                'verify_acc': valid_acc_12e,
                'final_test_acc': final_test_acc
            })
        except Exception as e:
            print(f"Error querying table during verification phase: {e}")

    # Re-rank based on 12-Epoch validation accuracy
    final_champions = sorted(verified_results, key=lambda x: x['verify_acc'], reverse=True)

    print("\nFinal Two-Stage Ranked Leaderboard")
    for final_rank, res in enumerate(final_champions, 1):
        mark = "*" if final_rank == 1 else " "
        print(
            f"{mark} Final Rank {final_rank:2d} | Original SCS Rank: {res['scs_rank']:2d} | 12-Epoch Valid Acc: {res['verify_acc']:5.2f}% | Final True Test Acc: {res['final_test_acc']:5.2f}% | Arch: {res['arch']}")

    # Save SOTA architecture and write to log
    best_structure_str = final_champions[0]['arch']
    best_test_acc = final_champions[0]['final_test_acc']
    best_valid_12e = final_champions[0]['verify_acc']

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)

    logging.info("=========== Final Lookup Results (Two-Stage NAS) ===========")
    logging.info(f"SOTA Architecture from 12-Epoch Verification: {best_structure_str}")
    logging.info(f"12-Epoch Validation Accuracy: {best_valid_12e:.2f}%")
    logging.info(f"200-Epoch True Test Accuracy: {best_test_acc:.2f}%")
    logging.info("=================================================")

    print("\nTwo-Stage evolutionary search successfully completed.")
    print(f"True test accuracy of the final SOTA architecture: {best_test_acc:.2f}%")