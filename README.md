# AEST-NAS

**Agentic Evolution with Semantic Topology-Aware Reranking for Training-Free Neural Architecture Search**

AEST-NAS is a training-free evolutionary neural architecture search (NAS) framework that separates **proxy-guided numerical exploration** from **semantic topology-aware reranking**. Zero-cost scores provide the repeated fitness signal during search, while large language models (LLMs) are restricted to constrained mutation and a final structural audit of a small elite set.

> **Repository status:** This is a deliberately trimmed public implementation. It includes the released search entry points, deterministic seed utilities, and the reported NAS-Bench-201 configuration, but excludes private development modules, raw provider transcripts, and the full internal experiment archive.

## Method at a glance

```text
Dataset + search-space specification
                |
                v
Search-space-conditioned directive compiler
        |                           |
        v                           v
Mutation directive          Semantic-audit directive
        |                           |
        v                           |
Phase 1: Agentic Evolution          |
LLM mutation + zero-cost fitness    |
        |                           |
        v                           |
Proxy-screened Top-K candidates ----+
                |
                v
Phase 2: Semantic Tribunal
                |
                v
Selected architecture
```

The framework has three main components:

1. **Search-space-conditioned directive compilation.** Dataset and search-space descriptions are converted into separate mutation and semantic-audit directives.
2. **Bounded-context Agentic Evolution.** The LLM proposes constrained mutations using localized population information, while ZiCo or another zero-cost proxy supplies the numerical fitness signal.
3. **Semantic topology-aware reranking.** The final elite set is compared using a shared capacity-flow-dead operation abstraction together with topology- and position-dependent structural criteria.

The proxy score and semantic judgment are intentionally not combined into a single scalar objective. Phase 1 raises the quality ceiling of the returned candidate set; Phase 2 reduces the risk of selecting a misleading proxy maximum.

## Experimental coverage

AEST-NAS is evaluated on three architecture paradigms:

- **NAS-Bench-201:** a tabular four-node cell space containing 15,625 architectures;
- **DARTS:** a substantially larger cell-based search space;
- **MobileNetV2-based macro search:** a sequential inverted-residual search space under FLOP constraints.

### Main search results

| Search space | Dataset / budget | AEST-NAS result | Reporting protocol |
|---|---:|---:|---|
| NAS-Bench-201 | CIFAR-10 | 94.29 +/- 0.07% test accuracy | Five runs |
| NAS-Bench-201 | CIFAR-100 | 73.32 +/- 0.21% test accuracy | Five runs |
| NAS-Bench-201 | ImageNet16-120 | 46.38 +/- 0.33% test accuracy | Five runs |
| DARTS | CIFAR-10 | 2.40 +/- 0.14% test error | Five runs |
| DARTS | CIFAR-100 | 16.88 +/- 0.09% test error | Five runs |
| MobileNetV2 macro | 398.1M FLOPs | 21.4% ImageNet Top-1 error | Single-run case study |
| MobileNetV2 macro | 548.3M FLOPs | 19.7% ImageNet Top-1 error | Single-run case study |

These results are reported as evidence of competitive search performance across different architecture representations. They are not presented as a claim of uniform statistical superiority over every baseline.

### Controlled Phase 1 comparison

The Phase 1 experiment compares LLM-guided mutation (LLM-PM), parameter-matched random mutation (RandPM), and random search under equal evaluation budgets. `Best@10` is a post-hoc measure of candidate-pool quality; benchmark validation and test accuracy are not used during search.

| Dataset | Mean Best@10 accuracy | Best observed | Gain over stronger equal-budget baseline |
|---|---:|---:|---:|
| CIFAR-10 | 94.22 +/- 0.27% | 94.37% | +0.24 points |
| CIFAR-100 | 72.96 +/- 0.48% | 73.51% | +1.39 points |
| ImageNet16-120 | 46.59 +/- 0.17% | 46.71% | +0.30 points |

Each value is computed over the fixed search seeds `0, 1, 2, 3, 4`, with a budget of 300 proxy evaluations per seed. The exact released seed derivation and reported settings are recorded in [`configs/nasbench201_reported.json`](configs/nasbench201_reported.json).

### Controlled Phase 2 comparison

Phase 2 is evaluated independently of Phase 1. For each dataset, five pools of 200 uniformly sampled, non-duplicate NAS-Bench-201 architectures are scored by ZiCo and reduced to proxy Top-10 sets. The Semantic Tribunal receives candidate identifiers and genotypes but not numerical proxy scores or benchmark accuracies. Candidate identifiers preserve descending proxy order, so the protocol is score-blind and accuracy-blind but not order-blind.

| Dataset | Tribunal selected test accuracy | Regret@10 to pool oracle |
|---|---:|---:|
| CIFAR-10 | 93.76 +/- 0.24% | 0.01 +/- 0.01 points |
| CIFAR-100 | 71.12 +/- 0.72% | 0.06 +/- 0.13 points |
| ImageNet16-120 | 45.49 +/- 0.72% | 0.02 +/- 0.05 points |

`Oracle@10` is used only as a post-hoc upper bound. These results concern small proxy-screened candidate sets and should not be interpreted as evidence that the Semantic Tribunal is an accuracy oracle.

## Reference configuration used in the manuscript

The tables below record the configuration used for the reported experiments. They should be treated as the reproduction target. Some defaults in the current public scripts reflect earlier development runs and therefore differ from the manuscript settings.

### Search and selection settings

| Item | NAS-Bench-201 | DARTS | MobileNetV2 macro |
|---|---:|---:|---:|
| Initial population | 51 | 50 | 50 |
| Maximum population | 100 | 100 | 100 |
| Default zero-cost proxy | ZiCo | ZiCo | ZiCo |
| Candidates passed to Phase 2 (`K`) | 10 | 20 | 20 |
| Typical completed search length | 100-300 iterations | 300-600 iterations | 891 iterations |
| Independent runs | 5 | 5 | 1 case-study run |
| Reported metric | Benchmark test accuracy | Trained test error | ImageNet Top-1 error |

For the controlled Phase 1 ablation, every method receives **300 proxy evaluations per seed**, uses the same population limits, and is repeated with five independent seeds. The NAS-Bench-201 initialization count is 51 because this matches the executed loop boundary. For the controlled Phase 2 ablation, each of the 15 independent pools contains 200 uniformly sampled, non-duplicate NAS-Bench-201 architectures; ZiCo retains the Top-10 candidates before reranking.

### LLM settings

| Role | Model used in the manuscript | Temperature | Invocation pattern |
|---|---|---:|---|
| Directive compilation | Gemini 2.5 Pro | Provider default unless otherwise specified | Once per search-space/dataset configuration |
| Phase 1 constrained mutation | DeepSeek-Chat | 1.0 | Once per completed search iteration |
| Phase 2 semantic reranking | DeepSeek-R1 | 0.0 | Once for the final Top-K candidate set |

Typical token usage was approximately 850-1,500 input tokens and 50-800 output tokens for a Phase 1 call, and 600-1,000 input tokens plus 1,000-1,500 output tokens for the single Phase 2 call. Provider model names, endpoints, and prices may change; record the exact provider-side model identifier and access date in any reproduction report.

`Agentic_NAS201.py` exposes the seed, proxy base seed, model identifiers, temperatures, prompt paths, population settings, and stopping parameters on the command line. Its defaults now match the NAS-Bench-201 configuration above; provider-side model revisions should still be recorded for every new run.

### Final evaluation protocols

#### DARTS architectures

| Parameter | Value |
|---|---:|
| Initial channels | 36 |
| Layers | 20 |
| Training epochs | 600 |
| Batch size | 96 |
| Optimizer | SGD |
| Initial learning rate | 0.025 |
| Momentum | 0.9 |
| Weight decay | `3e-4` |
| Cutout length | 16 |
| Gradient clipping | 5 |
| Path dropout probability | 0.2 |
| Auxiliary-tower weight | 0.4 |

#### MobileNetV2 macro architectures on ImageNet

| Parameter | Value |
|---|---:|
| Training epochs | 250 |
| Total batch size | 1,024 |
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Weight decay | `4e-5` |
| Initial learning rate | 0.4 |
| Learning-rate schedule | Cosine decay |
| Data augmentation | Random resized crop and horizontal flip |
| Label smoothing | 0.1 |
| Nominal search budgets | 450M and 600M FLOPs |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Violet-Evgarden/AEST-NAS.git
cd AEST-NAS
```

### 2. Create an environment

```bash
conda create -n aest-nas python=3.9
conda activate aest-nas
pip install -r requirements.txt
pip install python-dotenv
```

`python-dotenv` is required by `evolution_search.py` but is not yet listed in the snapshot's `requirements.txt`.

### 3. Install PyTorch

Install the build appropriate for your CUDA environment. The reported experiments used PyTorch 2.5.1 with CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for other CUDA versions or CPU-only environments.

### 4. Install NAS-Bench-201 support

```bash
pip install git+https://github.com/D-X-Y/NAS-Bench-201.git
pip install git+https://github.com/D-X-Y/xautodl.git
```

Download `NAS-Bench-201-v1_1-096897.pth` from the official NAS-Bench-201 release and place it in the repository root, or update the API path in the relevant scripts.

## Configuration before running

The public code is a research snapshot and retains several environment-specific settings. Review these items before starting a search:

1. Set the LLM API key through an environment variable; do not place a real key in source code:

   ```bash
   # Linux / macOS
   export DEEPSEEK_API_KEY="your-key"

   # Windows PowerShell
   $env:DEEPSEEK_API_KEY="your-key"
   ```

2. Pass the NAS-Bench-201 `.pth` location through `--api_path`.
3. Pass local prompt templates through `--phase1_prompt_path` and `--phase2_prompt_path`.
4. Confirm the intended provider model identifiers for each LLM call.
5. Check dataset paths, output directories, GPU indices, and FLOP constraints for your environment.

The trimmed public repository does not contain the paper's final prompt templates. Provide local templates that match the documented input/output fields, and record the exact prompt hash and provider model identifier for every new run. No API keys, benchmark databases, private datasets, or deleted development modules are included.

## Example: NAS-Bench-201 search

After completing the configuration above, a NAS-Bench-201 search can be started with:

```bash
python Agentic_NAS201.py \
  --gpu 0 \
  --dataset cifar100 \
  --api_path /path/to/NAS-Bench-201-v1_1-096897.pth \
  --seed 0 \
  --proxy_base_seed 20260722 \
  --evolution_max_iter 300 \
  --initial_random 51 \
  --population_size 100 \
  --save_dir ./output/cifar100
```

Available dataset values are `cifar10`, `cifar10-valid`, `cifar100`, and `ImageNet16-120`. The two CIFAR-10 spellings share the released `cifar10` seed derivation and use the `cifar10-valid` NAS-Bench-201 API split.

## Command-line reference

### `Agentic_NAS201.py`

This is the main NAS-Bench-201 search entry point in the current snapshot.

| Argument | Type / choices | Script default | Meaning |
|---|---|---:|---|
| `--gpu` | integer | `0` | CUDA device index used for zero-cost scoring |
| `--dataset` | `cifar10`, `cifar10-valid`, `cifar100`, `ImageNet16-120` | `cifar100` | NAS-Bench-201 dataset split |
| `--api_path` | path | `./NAS-Bench-201-v1_1-096897.pth` | Local benchmark API file |
| `--seed` | integer | `0` | Reported run seed; the controlled NAS-Bench-201 runs use `0`--`4` |
| `--proxy_base_seed` | integer | `20260722` | Base value for architecture-specific proxy seeds |
| `--evolution_max_iter` | integer | `300` | Maximum number of evolutionary iterations |
| `--initial_random` | integer | `51` | Initial uniformly sampled evaluations |
| `--batch_size` | integer | `32` | Batch size used to estimate the zero-cost proxy |
| `--population_size` | integer | `100` | Population capacity maintained by the search |
| `--top_k` | integer | `10` | Candidates sent to semantic reranking |
| `--patience_start` | integer | `200` | Earliest iteration at which stagnation is counted |
| `--patience` | integer | `50` | Non-improving evaluations before early termination |
| `--phase1_model` | string | `deepseek-chat` | Mutation model identifier |
| `--phase2_model` | string | `deepseek-reasoner` | Semantic-reranking model identifier |
| `--phase1_temperature` | float | `1.0` | Mutation sampling temperature |
| `--phase2_temperature` | float | `0.0` | Reranking sampling temperature |
| `--save_dir` | path | `./output` | Directory for search outputs |

The script writes `run_config.json`, `phase2_candidates.json`, and `phase2_selection.json` for each run. A malformed Tribunal response now terminates the run instead of silently selecting candidate 0.

### `evolution_search.py`

This entry point targets the MobileNetV2-based macro search space.

| Argument | Type | Script default | Meaning |
|---|---|---:|---|
| `--gpu` | integer | `0` | CUDA device index |
| `--zero_shot_score` | string | `Zen` | Zero-cost fitness name; the manuscript configuration uses ZiCo |
| `--search_space` | path | `SearchSpace/search_space_IDW_fixfc.py` | Search-space definition |
| `--evolution_max_iter` | integer | `1000` | Maximum evolutionary iterations |
| `--budget_model_size` | float / none | `None` | Optional model-size constraint |
| `--budget_flops` | float / none | `None` | Optional FLOP constraint |
| `--budget_latency` | float / none | `None` | Optional latency constraint |
| `--max_layers` | integer / none | `None` | Optional maximum network depth |
| `--batch_size` | integer | `32` | Proxy-evaluation batch size |
| `--input_image_size` | integer | `32` | Input resolution used by the script |
| `--population_size` | integer | `512` | Population capacity |
| `--save_dir` | path | `./output` | Output directory |
| `--gamma` | float | `1e-2` | Search-space scaling/control parameter |
| `--num_classes` | integer | `10` | Number of output classes |

The macro-search script retains development defaults and must be aligned with the intended dataset, FLOP budget, proxy, and image resolution before reproduction. In particular, do not use the default `input_image_size=32` for an ImageNet final evaluation.

### `test_proxy_batch.py`

This utility scores a manually supplied NAS-Bench-201 architecture list and retrieves its tabular benchmark performance.

| Argument | Choices | Script default |
|---|---|---:|
| `--gpu` | CUDA device index | `2` |
| `--proxy` | `SCS`, `ISR`, `ZES`, `Zen`, `ZiCo`, `TE-NAS`, `Syncflow`, `GradNorm`, `NASWOT` | `Syncflow` |
| `--dataset` | `cifar10`, `cifar100`, `ImageNet16-120` | `cifar10` |

The `candidate_archs` list in the current file is intentionally empty. Populate it with valid NAS-Bench-201 genotypes before running the utility. Benchmark accuracy is retrieved only for analysis and is not fed back into AEST-NAS search or reranking.

## Repository structure

```text
AEST-NAS/
├── Agentic_NAS201.py              # NAS-Bench-201 agentic search entry point
├── reproducibility.py             # Deterministic seed and run-metadata helpers
├── configs/
│   └── nasbench201_reported.json  # Reported NAS-Bench-201 settings
├── evolution_search.py            # MobileNetV2-based macro search
├── test_proxy_batch.py            # Batch proxy and benchmark lookup utility
├── train_image_classification.py  # Image classification training entry point
├── ZeroShotProxy/                 # Zero-cost proxy implementations
├── SearchSpace/                   # Macro search-space definitions
├── DataLoader/                    # Dataset loading utilities
├── ModelLoader/                   # Model construction and loading utilities
├── PlainNet/                      # PlainNet representation and operators
├── descriptions/                  # Search-space and proxy descriptions
├── mobile/                        # Mobile training utilities
└── requirements.txt
```

Additional files such as `evolution_201.py` are retained exploratory or legacy utilities and are not the primary entry points for the revised AEST-NAS workflow.

## Reproducibility checklist

The trimmed public repository records:

- fixed reported search seeds `0`--`4` and deterministic BLAKE2b seed derivation;
- architecture-specific, method- and order-independent zero-cost proxy seeding;
- the reported NAS-Bench-201 population, budget, LLM, and aggregation settings;
- per-run configuration, Phase-2 candidate lists, and raw selection output for newly executed searches;
- the fact that Phase-2 candidate order preserves descending proxy rank.

The full internal experiment archive, discarded development modules, API credentials, benchmark database, and raw historical provider transcripts are intentionally not included. This repository should therefore be cited as the released implementation and configuration record, not as a byte-for-byte archive of every development run.

## API and compute cost

Across the evaluated configurations, the observed end-to-end API expenditure was approximately **US$0.34-US$0.97 per search task**. The value depends on provider pricing, model selection, prompt length, network latency, and the number of completed iterations. Reported GPU-day values should also be interpreted as approximate because hardware and accounting protocols differ across studies.

## Citation

The revised manuscript is currently being prepared for submission. A complete BibTeX entry will be added after a public paper record becomes available. In the meantime, please cite this repository by URL and include the commit hash used in your experiments.

## Contact

For questions about the code or experiments, please open a GitHub issue or contact the corresponding author listed in the manuscript.
