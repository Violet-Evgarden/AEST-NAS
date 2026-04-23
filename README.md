# 🚀 AEST-NAS
**Agentic Evolution Meets Semantic Tribunal: A Self-Designing Dual-Phase Framework for Neural Architecture Search**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Paper_Title-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

> This is the official PyTorch implementation of **AEST-NAS**. 
## 📖 Introduction
Traditional Zero-Cost Neural Architecture Search (NAS) often suffers from the **"Proxy Gap"** and **"Skip-Connect Collapse"**, where proxy metrics yield false-positive high scores for degenerate architectures. 

**AEST-NAS** introduces a novel self-designing dual-phase framework to conquer this. By leveraging a **Meta-LLM** to autonomously extract physical topological rules, our framework eliminates human-engineered prompts. It seamlessly integrates a fast **Agentic Evolution (Phase 1)** for broad exploration with a rigorous **Semantic Tribunal (Phase 2)** for zero-shot, proxy-free structural evaluation.

## 🌟 Key Features
- 🤖 **Self-Designing Meta-Agent:** Automatically generates architectural rules, semantic priors, and evaluation prompts directly from the search space definitions.
- 🧬 **Phase 1 - Agentic Evolution:** A dynamic swarm search loop powered by LLM reflection, utilizing both Long-term (historical failed attempts) and Short-term (peer reference) memory.
- ⚖️ **Phase 2 - Semantic Tribunal:** A strict LLM judge that evaluates elite candidates *purely based on structural physics and topological semantics*, effectively rejecting false-positive architectures without relying on numerical proxy scores.

## 🏛️ Architecture
![AEST-NAS Architecture](docs/architecture.png)
*(Please ensure the image is placed in the `docs/` folder or update this path)*

## 📰 News & Updates
- **[2026.04]** 🚀 Repository created. Training and Search code will be released soon!
- **[2026.XX]** 📄 Paper submitted to [Target Conference].

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/Violet-Evgarden/AEST-NAS.git](https://github.com/Violet-Evgarden/AEST-NAS.git)
cd ASTRA-NAS

# Create a conda environment
conda create -n aest-nas python=3.9
conda activate aest-nas

# Install dependencies
pip install -r requirements.txt
