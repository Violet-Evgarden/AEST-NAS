# 🚀 AEST-NAS
**Agentic Evolution Meets Semantic Tribunal: A Meta-Prompted Dual-Phase Framework for Neural Architecture Search**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Paper_Title-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

> This is the official PyTorch implementation of **AEST-NAS**. 

## 📖 Introduction
While Zero-Cost (ZC) proxies exponentially accelerate Neural Architecture Search (NAS), they frequently exhibit severe mathematical biases, artificially elevating structurally pathological architectures (e.g., severe capacity deficits or degenerate connectivity). 

**AEST-NAS** introduces a novel meta-prompted dual-phase framework to decouple search efficiency from proxy biases. By leveraging a **Meta-Prompt Synthesizer** to autonomously translate universal structural heuristics into space-specific directives, our framework enables an efficient **Agentic Evolution (Phase 1)** guided by black-box ZC metrics, seamlessly safeguarded by a **Semantic Tribunal (Phase 2)** that performs domain-aware semantic filtering.

## 🌟 Key Features
- 🤖 **Meta-Prompt Synthesizer:** Autonomously translates well-established structural heuristics (e.g., Path Connectivity, Representational Density) into explicit, space-specific directives, eliminating the need for manual prompt engineering across different search spaces.
- 🧬 **Phase 1 - Agentic Evolution:** An LLM-driven mutator agent that explores topologies using purely black-box fitness metrics. It utilizes a stateless, short-term duplicate-aware feedback mechanism to prevent context pollution and topological hallucinations.
- ⚖️ **Phase 2 - Semantic Tribunal:** A strict structural adjudicator grounded in the **Semantic Structural Prior**. It evaluates the top-$K$ elite candidates to systematically filter out pathological outliers that exploit mathematical proxy formulations, selecting the most robust architecture for deployment.
- ⚡ **Extreme Efficiency & Low Cost:** Discovers highly compact optimal architectures (e.g., 19.7% Top-1 error on ImageNet under 600M FLOPs) with an end-to-end search cost of only **0.2 GPU days** and an API expenditure ranging from **$0.34 to $0.97**.

## 📊 Main Results
AEST-NAS achieves state-of-the-art Pareto-optimal trade-offs between accuracy and search efficiency across multiple search spaces:
- **NAS-Bench-201:** 94.29% Test Acc (CIFAR-10) | 73.32% Test Acc (CIFAR-100)
- **DARTS Space:** 2.40% Error (CIFAR-10) | 16.88% Error (CIFAR-100)
- **MobileNet Space (ImageNet):** 19.7% Top-1 Error (600M FLOPs constraint)

## 🏛️ Architecture
![AEST-NAS Architecture](docs/architecture.png)
*(Note: System workflow detailing the Meta-Prompt Synthesizer and the Dual-Phase Search Loop)*

## 📰 News & Updates
- **[2026.05]** 📄 Paper submitted to Expert Systems with Applications (ESWA).
- **[2026.04]** 🚀 Repository created. Search and training codes are progressively being open-sourced.

## ⚙️ Installation

**Step 1: Environment Setup**
We recommend using Conda to create a clean virtual environment for AEST-NAS.

```bash
# Clone the repository
git clone https://github.com/Violet-Evgarden/AEST-NAS.git
cd AEST-NAS

# Create and activate a conda environment
conda create -n aest-nas python=3.9
conda activate aest-nas
```

**Step 2: Install Basic Dependencies**
Install the standard scientific and auxiliary libraries required for the project:

```bash
pip install -r requirements.txt
```

**Step 3: Install PyTorch**
Please install PyTorch, TorchVision, and TorchAudio according to your specific hardware configuration. 
*Note: Our experiments were conducted using NVIDIA RTX 4090 GPUs. If you are using a similar setup with CUDA 12.1, you can install PyTorch using the following command:*

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
For other CUDA versions or CPU-only installations, please refer to the [PyTorch Official Website](https://pytorch.org/get-started/locally/).

**Step 4: Install Architecture Search APIs (NAS-Bench-201 & xautodl)**
To properly load the search space and evaluate the network architectures, you need to install the official NAS-Bench-201 API and the `xautodl` library. Please install them directly from their source repositories:

```bash
pip install git+https://github.com/D-X-Y/NAS-Bench-201.git
pip install git+https://github.com/D-X-Y/xautodl.git
```
