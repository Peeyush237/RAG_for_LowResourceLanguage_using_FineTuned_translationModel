<div align="center">

# IndicTrans2: Bidirectional English-Odia Fine-tuning
### Parameter-Efficient Translation with LoRA Adapters

## This project's Readme.md will be updated accordingly as the project progresses.
<br>
</div>

---

## Overview

Fine-tuned **IndicTrans2 1B** models for **bidirectional** English-Odia translation using **LoRA (Low-Rank Adaptation)**. The project ships ready-to-use LoRA adapters for both directions, trained on custom parallel corpora and evaluated against standard MT benchmarks.

---

## Results

| Direction | Base Model | BLEU Score |
| :--- | :--- | :---: |
| **English to Odia** | `ai4bharat/indictrans2-en-indic-1B` | **33** |
| **Odia to English** | `ai4bharat/indictrans2-indic-en-1B` | **47** |

---

## Repository Structure

```
.
├── indictrans2-odia-bidirectional.ipynb   # Training & evaluation notebook
├── it2_en2or_lora_adapter/               # Fine-tuned LoRA adapter (En → Or)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── dict.SRC.json / dict.TGT.json
│   ├── model.SRC / model.TGT
│   └── tokenizer_config.json
├── it2_or2en_lora_adapter/               # Fine-tuned LoRA adapter (Or → En)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── dict.SRC.json / dict.TGT.json
│   ├── model.SRC / model.TGT
│   └── tokenizer_config.json
└── README.md
```

---

## Training Configuration

### English to Odia (`it2_en2or_lora_adapter`)

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `ai4bharat/indictrans2-en-indic-1B` |
| **LoRA Rank (r)** | 8 |
| **LoRA Alpha** | 16 |
| **LoRA Dropout** | 0.1 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` |
| **Task Type** | `SEQ_2_SEQ_LM` |
| **Precision** | `fp16` (Mixed Precision) |
| **Hardware** | Single T4 GPU (Google Colab) |

### Odia to English (`it2_or2en_lora_adapter`)

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `ai4bharat/indictrans2-indic-en-1B` |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| **Task Type** | `SEQ_2_SEQ_LM` |
| **Precision** | `fp16` (Mixed Precision) |
| **Hardware** | Single T4 GPU (Google Colab) |

---

## Pipeline

1. **Environment Setup** -- Install `IndicTransToolkit`, pinned `transformers`, `peft`, `torch`, and `accelerate`.
2. **Dataset Preparation** -- Parse tab-separated parallel corpus, apply Odia script validation, create train/eval splits.
3. **Baseline Evaluation** -- Zero-shot BLEU & chrF++ on a held-out validation set.
4. **LoRA Fine-tuning** -- Seq2Seq training with gradient accumulation for both translation directions.
5. **Adapter Export** -- Save LoRA weights as standalone adapters for lightweight deployment.

---

## Requirements

```bash
numpy >= 2.1
torch >= 2.5
transformers == 4.38.2
accelerate == 0.27.2
peft >= 0.10.0
indictranstoolkit
sacrebleu
sentencepiece
```

> **Note:** Restart the runtime after the initial installation cell so the compiled `IndicProcessor` is correctly registered.

---

## Evaluation Metrics

- **BLEU** -- Measures n-gram precision between predicted and reference translations.
- **chrF++** -- Character n-gram F-score; particularly effective for morphologically rich languages like Odia.

---

## Quick Start (Inference)

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- English to Odia ---
base_en2or = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
model_en2or = PeftModel.from_pretrained(base_en2or, "it2_en2or_lora_adapter")

# --- Odia to English ---
base_or2en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B")
model_or2en = PeftModel.from_pretrained(base_or2en, "it2_or2en_lora_adapter")
```

---

<div align="center">
  <b>Bidirectional. Parameter-efficient. Script-faithful.</b>
</div>
