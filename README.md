# 🌟 CLEAR: Rethinking Hallucinations in Multimodal Large Language Models as Causal Misattribution
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> This is the official repository for the paper **"CLEAR: Rethinking Hallucinations in Multimodal Large Language Models as Causal Misattribution"**.

---

## 📖 Abstract

Multimodal Large Language Models (MLLMs) are prone to object hallucinations, generating visually unsupported content that undermines reliability. We frame this issue from a causal perspective, attributing hallucinations to spurious visual co-occurrence bias and over-reliance on autoregressive language priors.
To address this, we propose **CLEAR** (**C**ausal **L**atent **E**xtraction **A**nd **R**e-normalization), a unified framework that mitigates hallucinations during both training and inference. During training, CLEAR employs a text-guided Latent Causal Disentanglement (LCD) module with in-image counterfactual intervention to separate object-relevant features from confounding background context. During inference, we introduce Mask-Guided Contrastive Decoding (MGCD), which contrasts factual and counterfactual decoding paths to suppress language-prior-driven errors.
We further perform Probability-Space Intervention (PSI) in the normalized attention space, enabling stable intervention while preserving spatial structure. Extensive experiments show that CLEAR consistently reduces hallucinations across benchmarks while maintaining strong multimodal reasoning performance, demonstrating the effectiveness of causal intervention for improving model faithfulness.

---

## 🎯 Causal Modeling of Object Hallucination Process

<p align="center">
  <img src="images/teaser.png" width="80%">
</p>

We reveal that hallucinations are essentially **causal misattributions** triggered by two erroneous paths:

1. **Language Prior Bias ($X_{text} \rightarrow Y$)**
2. **Spurious Co-occurrence Bias ($Z_{bg} \rightarrow Y$)**  

<p align="center">
  <img src="images/framework.png" width="90%">
</p>

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone CLEAR.git
cd CLEAR
````

### 2. Set Up the Environment

```bash
conda create -n clear python=3.10 -y
conda activate clear
pip install --upgrade pip
pip install -e .
```

---

## 🧩 Pre-requisites

### 1. CLIP Model

Download the vision encoder:

* HuggingFace: `clip-vit-large-patch14-336`

### 2. Vicuna Base Model

Download the language backbone:

* HuggingFace: `lmsys/vicuna-7b-v1.5`

Place them into:

```bash
./checkpoints/
```

### 3. Evaluation Datasets

Please follow the official LLaVA evaluation instructions to download:

* POPE
* MSCOCO (for CHAIR)
* MME
* MMBench

Put them under:

```bash
./playground/data/eval/
```

---

## 🚀 Training (Latent Causal Disentanglement)

```bash
bash scripts/v1_5/finetune_lora.sh
```

> ⚠️ If using LLaMA-2 backbone, change:

```bash
--version vicuna_v1 → --version v1
```

---

## 🧪 Inference & Evaluation (MGCD + PSI)

### Evaluate on POPE

```bash
bash run_eval_pope.sh
```

### Evaluate on MME

```bash
bash run_eval_mme.sh
```

### Evaluate on CHAIR

```bash
bash run_eval_chair.sh
```

---

## 💡 Hyperparameters

Modify in:

```bash
llava/eval/model_vqa_loader.py
```

* `threshold (T)` → visual attention threshold (default: 0.20)
* `alpha (λ)` → contrastive strength (default: 0.4)
* `start_layer / end_layer` → intervention range (default: 8–28)

---

## 🏆 Main Results

<p align="center">
  <img src="images/main_results.png" width="90%">
</p>

CLEAR consistently achieves:

* ✅ **Best hallucination suppression (POPE / CHAIR)**
* ✅ **No degradation in cognitive ability (MMBench / MME)**
* ✅ **Strong generalization across model scales**

---

## 🙏 Acknowledgement

This project builds upon:

* LLaVA: Large Language and Vision Assistant
* Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution
* VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
* OPEAR: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation

We sincerely thank the open-source community for their contributions.

---
