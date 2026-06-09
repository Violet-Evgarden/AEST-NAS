import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import PlainNet
from nas_201_api import NASBench201API as API
from nas_201_api.models import get_cell_based_tiny_net

# Load the NAS-Bench-201 database globally
print("Loading NAS-Bench-201 database, please wait...")
nas201_api = API('./NAS-Bench-201-v1_1-096897.pth')
print("Loading complete.")

from openai import AzureOpenAI
from dotenv import load_dotenv

from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, \
    compute_NASWOT_score
import benchmark_network_latency

working_dir = os.path.dirname(os.path.abspath(__file__))


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--zero_shot_score', type=str, default='Zen',
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default='SearchSpace/search_space_IDW_fixfc.py',
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(1000),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None,
                        help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None,
                        help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None,
                        help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=32, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=32,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


# Inject num_replaces parameter into the Prompt
def generate_by_llm(structure_str, score, num_replaces, top_3=None, bottom_3=None):
    if top_3 is None: top_3 = []
    if bottom_3 is None: bottom_3 = []

    file_path = "prompt/template.txt"
    import os
    from openai import OpenAI

    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt.replace("{{architecture}}", structure_str)
    prompt = prompt.replace("{{score}}", str(score))

    # Enforce mutation intensity instruction
    mutation_instruction = f"\n\n=== MUTATION INTENSITY (CRITICAL) ===\n"
    mutation_instruction += f"You MUST strictly mutate EXACTLY {num_replaces} component(s) from the original architecture.\n"
    if num_replaces == 1:
        mutation_instruction += "Make a VERY SMALL change (e.g., change ONLY ONE kernel size, OR ONE channel width).\n"
    else:
        mutation_instruction += "Make a LARGER change (e.g., modify multiple blocks, or change depth and width simultaneously).\n"
    prompt += mutation_instruction

    # Inject dual-track memory pool
    memory_injection = "\n\n=== DUAL-TRACK EXPERIENCE POOL ==="

    if len(top_3) > 0:
        memory_injection += "\n[CURRENT PROMISING ARCHITECTURES - Consider these for inspiration but keep exploring]:\n"
        for i, record in enumerate(top_3):
            memory_injection += f"Rank {i + 1}: Score={record['score']:.4f} | Arch={record['arch']}\n"
    if len(bottom_3) > 0:
        memory_injection += "\n[TABOOS - Avoid these low-scoring or invalid architectures]:\n"
        for i, record in enumerate(bottom_3):
            memory_injection += f"Worst {i + 1}: Score={record['score']:.4f} | Arch={record['arch']}\n"

    prompt += memory_injection

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        stop=["<|im_end|>"]
    )
    new_structure_str = response.choices[0].message.content
    return new_structure_str


def get_new_random_structure_str(AnyPlainNet, structure_str, score, num_classes, get_search_space_func,
                                 num_replaces, top_3=None, bottom_3=None):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)

    get_new_llm_structure_str = generate_by_llm(structure_str, score, num_replaces, top_3=top_3, bottom_3=bottom_3)

    return get_new_llm_structure_str


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str


def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
                                                              resolution=args.input_image_size,
                                                              in_channels=3, gpu=gpu, repeat_times=1,
                                                              fp16=True)
    del the_model
    torch.cuda.empty_cache()
    return the_latency


def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=True)
    the_model = the_model.cuda(gpu)
    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
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

        elif args.zero_shot_score == 'Flops':
            the_nas_core = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_core = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_core = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
    except Exception as err:
        logging.info(str(err))
        logging.info('--- Failed structure: ')
        logging.info(str(the_model))
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

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    # Core modification: Inject a valid initial seed architecture to prevent empty structures
    initial_structure_str = "SuperConvK3BNRELU(3,32,1,1)SuperResK3K3(32,64,2,64,1)SuperResK3K3(64,128,2,128,1)"
    print("[Initial Seed Revealed] The first architecture sent to the LLM is as follows:")
    print(initial_structure_str)

    initial_score = compute_nas_score(AnyPlainNet, initial_structure_str, gpu, args)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter):
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(
                f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time / 3600:4g}h')

        # Dynamically extract Top 3 and Bottom 3
        current_top_3 = []
        current_bottom_3 = []

        if len(popu_structure_list) > 0:
            history_records = [{"arch": arch, "score": s} for arch, s in
                               zip(popu_structure_list, popu_zero_shot_score_list)]
            history_records.sort(key=lambda x: x["score"], reverse=True)

            # Warm-up Phase: e.g., set the first 50 iterations as exploration phase to prevent premature convergence
            WARMUP_STEPS = 50

            if loop_count >= WARMUP_STEPS:
                current_top_3 = history_records[:3]
            else:
                current_top_3 = []

            current_bottom_3 = history_records[-3:] if len(history_records) >= 3 else history_records

        if len(popu_structure_list) <= 10:
            print(
                f"\n[Cold Start Phase] Generating architecture {len(popu_structure_list) + 1}/11 based on initial seed (Base score: {initial_score:.4f})...")

            taboo_list = [{"arch": arch, "score": score} for arch, score in
                          zip(popu_structure_list, popu_zero_shot_score_list)]
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=initial_structure_str, score=initial_score,
                num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=1,
                top_3=current_top_3, bottom_3=current_bottom_3)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            tmp_score = popu_zero_shot_score_list[tmp_idx]

            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=tmp_random_structure_str, score=tmp_score,
                num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=2,
                top_3=current_top_3, bottom_3=current_bottom_3)

        print("\nDeepSeek Raw Output:")
        print(random_structure_str)
        print("\n")

        # Parsing logic
        import re
        try:
            match = re.search(r'"arch"\s*:\s*"([^"]+)"', random_structure_str)
            if match:
                random_structure_str = match.group(1).replace(" ", "").replace("\n", "").replace("\\", "")

                # Anti-lazy duplication check (used only in evolution phase)
                if len(popu_structure_list) > 10 and random_structure_str == tmp_random_structure_str:
                    print("LLM generated a duplicate! Forcing mutation.")
                    random_structure_str = random_structure_str.replace("K3", "K5", 1)
                    if random_structure_str == tmp_random_structure_str:
                        random_structure_str = random_structure_str.replace("K5", "K3", 1)

                print("Extraction successful! Network structure:", random_structure_str)
            else:
                print("Extraction failed: 'arch' field not found.")
        except Exception as e:
            print("Extraction error:", e)

        random_structure_str = get_splitted_structure_str(AnyPlainNet, random_structure_str,
                                                          num_classes=args.num_classes)
        the_model = None

        # Validation
        if args.max_layers is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_layers = the_model.get_num_layers()
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue

        the_nas_core = compute_nas_score(AnyPlainNet, random_structure_str, gpu, args)

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

    # Export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)