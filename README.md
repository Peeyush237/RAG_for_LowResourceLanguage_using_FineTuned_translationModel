<div align="center">

# IndicTrans2: English to Odia Fine-tuning
### Robust Translation Pipeline with Native Script Enforcement

<br>
</div>

---

## Overview

This repository contains a comprehensive Jupyter Notebook pipeline for fine-tuning the **ai4bharat/indictrans2-en-indic-1B** model specifically for English-to-Odia translation. The core focus of this project is resolving the common "Devanagari fallback" bug where Odia translations incorrectly surface in the Devanagari script, ensuring native, accurate, and contextually rich Odiya script generation.

By leveraging **LoRA (Low-Rank Adaptation)**, this pipeline achieves efficient fine-tuning on consumer-grade hardware (like a T4 GPU) without sacrificing linguistic nuance or translation quality.

---

## Key Innovations

*   **Native Script Enforcement:** Integrates `IndicProcessor` and explicitly forces the `forced_bos_token_id='ory_Orya'` token to guarantee 100% native Odia script output.
*   **Dependency Stability:** Pins critical library versions (`transformers`, `torch`, `numpy`) to prevent cross-compatibility issues during the Cython build phase of the toolkit.
*   **Efficiency via LoRA:** Utilizes the `peft` library for parameter-efficient parameter updates, reducing memory footprint while maintaining state-of-the-art results.
*   **Built-in Diagnostics:** Features automated script-checking at the data loading phase to catch polluted datasets before training begins.

---

## Pipeline Architecture

The workflow consists of five distinct, reproducible phases:

1.  **Environment Setup:** Rigid dependency installation including `IndicTransToolkit`.
2.  **Dataset Preparation:** Parsing tab-separated data (`train.final`), applying safety checks, and creating train/eval splits.
3.  **Baseline Evaluation:** Establishing baseline capabilities by evaluating the zero-shot model against standard BLEU and chrF++ metrics.
4.  **Parameter-Efficient Training:** Fine-tuning the base 1B parameter model using a Seq2Seq Trainer orchestrated with LoRA adapters.
5.  **Artifact Generation:** Freezing and exporting the optimized LoRA weights for downstream inference.

---

## Requirements and Installation

To ensure stability, the notebook enforces absolute version constraints. Do not manually upgrade these packages during execution.

```bash
# Core Dependencies
numpy >= 2.1
torch >= 2.5
transformers == 4.38.2
accelerate == 0.27.2
peft == 0.10.0

# Natural Language Tooling
indictranstoolkit
sacrebleu
sentencepiece
```

**Important Note:** The runtime must be restarted immediately after the step 0 installation cell to ensure the newly compiled `IndicProcessor` is correctly registered in the system path.

---

## Technical Specifications

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `ai4bharat/indictrans2-en-indic-1B` |
| **Task Type** | `SEQ_2_SEQ_LM` |
| **Source Language** | `eng_Latn` |
| **Target Language** | `ory_Orya` |
| **Hardware** | Single T4 GPU (CUDA) |
| **Epochs** | 3 |
| **Batch Size** | 2 (with Gradient Accumulation = 8) |
| **Learning Rate** | `5e-5` |
| **LoRA Rank (r)** | 8 |
| **LoRA Alpha** | 16 |
| **Target Modules** | `q_proj`, `v_proj`, `k_proj`, `out_proj`, `fc1`, `fc2` |
| **Precision** | `fp16` (Mixed Precision) |

### Dataset Format

The expected dataset format is a `train.final` file, composed as a non-header, tab-separated value (TSV) file with three strict columns:
1.  Source Document/Domain
2.  English Text
3.  Odia Text

---

## Evaluation Metrics

The script utilizes standard machine translation metrics to validate performance improvements.

*   **BLEU (char-level)**: Measures exact overlaps.
*   **chrF++**: Captures morphological agreements and character n-gram overlaps, which is highly effective for morphologically rich Indian languages like Odia.

The baseline evaluation runs automatically on an isolated 300-row validation subset before any fine-tuning occurs to establish a strict comparison floor.

---

## This project's Readme.md will be updated accordingly as the project progresses.

<div align="center">
  <br>
  <b>Designed for efficiency, accuracy, and absolute script fidelity.</b>
</div>
