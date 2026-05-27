---
license: cc-by-nc-4.0
task_categories:
- visual-question-answering
- image-to-text
language:
- en
tags:
- vision-language
- multimodal
- visual-question-answering
- measurement-grounding
- raw-image
- camera-raw
- meas-xyz
- low-light
- hdr
- benchmark
- prsimvl
pretty_name: MeasL-Bench-V1
size_categories:
- 1K<n<10K
---

# MeasL-Bench-V1

**MeasL-Bench-V1** is the held-out benchmark released with PRSIMVL for measurement-grounded vision-language evaluation. It tests whether models can recover and reason over evidence that may be weakened or removed by RGB rendering.

- **Hugging Face**: [kepeng/MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1)
- **Project page**: <https://kepengxu.github.io/projects/prism-vl/>
- **Paper**: <https://arxiv.org/abs/2605.11727>

## At A Glance

| Item | Value |
|---|---|
| Examples | 2,183 matched question-answer records per split. |
| Splits | Meas.-XYZ (`raw`) and matched RGB (`rgb`). |
| Images | 3,812 release image files. |
| Task | Visual question answering with measurement-sensitive evidence. |
| License | CC BY-NC 4.0, non-commercial research and education. |

## Why This Benchmark Exists

MeasL-Bench targets observation-interface failures, not only generic VQA difficulty. It emphasizes low-illumination evidence recovery, HDR/exposure-sensitive grounding, visibility-sensitive queries, hallucination-sensitive prompts, and standard RGB-sufficient grounding slices for coverage.

The benchmark is suitable for comparing:

1. RGB-native VLMs, and
2. measurement-grounded models that operate on Meas.-XYZ or related sensor-derived inputs.

## Released Files

| File | Size | Description |
|---|---:|---|
| `test-raw-measl-bench.jsonl` | 2,183 rows | Main benchmark using the Meas.-XYZ path protocol. |
| `test-rgb-measl-bench.jsonl` | 2,183 rows | Matched RGB counterpart for controlled comparison. |
| `image/` | 3,812 files | Local image assets referenced by both JSONL files. |

## Path Convention

JSONL records use release-local image paths such as:

```text
eval_data/image/measl_bench_raw_000000.png
```

When this dataset directory is placed under a project root as `eval_data/`, those paths work directly. When reading from the Hugging Face dataset repository root, remove the leading `eval_data/` prefix, so the same example maps to:

```text
image/measl_bench_raw_000000.png
```


## Hugging Face Usage

```python
from datasets import load_dataset

dataset = load_dataset(
    "kepeng/MeasL-Bench-V1",
    data_files={
        "raw": "test-raw-measl-bench.jsonl",
        "rgb": "test-rgb-measl-bench.jsonl",
    },
)
```

## JSONL Schema

Each row follows the release pipeline schema used by `scripts/test_raw_eval_pipeline_opt/`.

| Field | Meaning |
|---|---|
| `messages` | Conversation-style prompt containing the user question. |
| `images` | One or more release-local image paths. |
| `answer` | Reference answer for evaluation. |
| `question_type` | Capability label from the taxonomy below. |

## Capability Taxonomy

| Label | Capability | Count |
|---|---|---:|
| CAG | Chromatic Attribute Grounding | 150 |
| NG | Numerosity Grounding | 150 |
| DSG | Descriptive Scene Grounding | 150 |
| HER | HDR Evidence Recovery | 150 |
| LER | Low-Illumination Evidence Recovery | 233 |
| STR | Scene Text Recognition | 150 |
| GVG | General Visual Grounding | 150 |
| CVR | Compositional Visual Reasoning | 150 |
| SRU | Spatial Relation Understanding | 150 |
| MSQ | Manner and State Queries | 150 |
| EAQ | Entity and Attribute Queries | 150 |
| DS | Discriminative Selection | 150 |
| AEI | Agent and Entity Identification | 150 |
| BVV | Binary Visual Verification | 150 |

Total: **2,183** examples.

## Evaluation Protocol

- Use identical question-answer records for Meas.-XYZ and RGB matched evaluation.
- Report BLEU, ROUGE-L, and optional LLM-as-judge accuracy.
- Keep the denominator fixed to 2,183 for overall reporting.
- Use the release entrypoint documented in [`../eval/README.md`](../eval/README.md).

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
DATASET=rgb MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

## License

This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You may use, share, and adapt the dataset for non-commercial research and educational purposes, provided that you give appropriate credit and cite the associated project or paper. Commercial use is not permitted without prior written permission from the authors.

## Recommended Citation Name

Use this name in tables and dataset references:

**MeasL-Bench-V1: Measurement-grounded Language-Vision Benchmark**
