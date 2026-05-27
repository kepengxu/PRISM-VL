<p align="center">
  <img src="assets/prsimvl_logo.svg" alt="PRSIMVL logo" width="156">
</p>

<h1 align="center">PRSIMVL</h1>

<p align="center">
  <strong>Allegory of the Cave: Measurement-Grounded Vision-Language Learning</strong>
</p>

<p align="center">
  <a href="https://kepengxu.github.io/">Kepeng Xu</a> · Li Xu · Gang He · Wenxin Yu
</p>

<p align="center">
  <a href="https://kepengxu.github.io/projects/prism-vl/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2605.11727">arXiv:2605.11727</a> ·
  <a href="https://openreview.net/forum?id=fsCtGojL2R">Synthetic RAW precursor</a> ·
  <a href="https://huggingface.co/datasets/kepeng/MeasL-Bench-V1">MeasL-Bench-V1</a> ·
  <a href="https://huggingface.co/datasets/kepeng/MeasL-150K-V1">MeasL-150K-V1</a> ·
  <a href="https://huggingface.co/kepeng/PRSIMVL-LoRA-V1">Weights</a> ·
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <img alt="Method" src="https://img.shields.io/badge/Method-PRSIMVL-111827">
  <img alt="Input" src="https://img.shields.io/badge/Input-Meas.--XYZ-0a7ea4">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-MeasL--Bench--V1-2ea44f">
  <img alt="Training" src="https://img.shields.io/badge/Training-MeasL--150K--V1-f0883e">
  <img alt="Backbone" src="https://img.shields.io/badge/Base-Qwen3--VL-1f6feb">
  <img alt="Adapters" src="https://img.shields.io/badge/LoRA-2B%20%7C%204B%20%7C%208B-8957e5">
</p>

**PRSIMVL** is a research release for asking a simple but under-tested question: when the RGB image has already lost sensor evidence, can a vision-language model reason better from measurement-domain observations?

PRSIMVL keeps the familiar Qwen3-VL training and inference workflow, but changes the visual interface from post-ISP RGB to RAW-derived **Meas.-XYZ** plus camera metadata. The release includes the benchmark, training corpus, evaluation pipeline, service demo, and LoRA checkpoints needed to reproduce the core findings.

<p align="center">
  <img src="assets/prsimvl_framework.png" alt="PRSIMVL framework" width="92%">
</p>

## The 30-Second Version

| What | Release |
|---|---|
| Core idea | Use RAW-derived Meas.-XYZ and capture metadata when RGB rendering clips, denoises, tone maps, or quantizes away evidence. |
| Benchmark | [MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1), 2,183 held-out matched examples over 14 measurement-sensitive capability slices. |
| Training data | [MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1), 152,517 instruction-tuning examples with 48,000 release images. |
| Model family | Qwen3-VL 2B, 4B, and 8B with released PRSIMVL LoRA adapters hosted on [Hugging Face](https://huggingface.co/kepeng/PRSIMVL-LoRA-V1). |
| Headline result | PRSIMVL-8B improves over RGB Qwen3-VL-8B by **+0.1074 BLEU**, **+0.1071 ROUGE-L**, and **+4.46 LLM-Judge points** on MeasL-Bench. |

## Start Here

| Goal | Entry point | What you get |
|---|---|---|
| Ask one image question | [`inference/README.md`](inference/README.md) | Start `swift deploy`, send a local image with `ask_service.py`, inspect the answer. |
| Run the benchmark | [`eval/README.md`](eval/README.md) | Evaluate Meas.-XYZ or matched RGB with the packaged wrapper. |
| Inspect benchmark data | [`eval_data/README.md`](eval_data/README.md) | Dataset card, taxonomy, schema, HF loading snippet, path rules. |
| Inspect training data | [`training_data/README.md`](training_data/README.md) | Dataset card, release contents, quality checks, registry aliases. |
| Understand release scope | [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md) | What is included, pruned, and expected as large artifacts. |

## Quick Start

Install the release snapshot:

```bash
git clone <repo-url> PRSIMVL
cd PRSIMVL
bash install_editable.sh
```

Run a dry-run check without launching model inference:

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh --dry-run
```

Run one PRSIMVL adapter on the default Meas.-XYZ benchmark split:

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

Large artifacts are expected at these release-local paths:

```text
eval_data/       # MeasL-Bench-V1 JSONL + image/
training_data/   # MeasL-150K-V1 JSONL + image/
exps/            # released LoRA adapters
```

Download the public data from Hugging Face: [MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) for evaluation and [MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) for training. Released LoRA weights are hosted at [kepeng/PRSIMVL-LoRA-V1](https://huggingface.co/kepeng/PRSIMVL-LoRA-V1); restore them under `exps/` before running adapter inference or full evaluation.

## Why Measurement Grounding Matters

RGB is a display-oriented product of an image signal processor. It is useful, compact, and familiar, but it may remove the evidence that a downstream model needs. PRSIMVL treats the camera measurement as a first-class observation: Meas.-XYZ preserves a linear, three-channel view derived from RAW measurements, and metadata supplies capture context such as ISO, exposure time, and aperture.

The examples below show low-illumination text cases where RGB rendering exposes misleading evidence while Meas.-XYZ keeps the answer region recoverable.


| Case | RGB Observation | Meas.-XYZ Observation |
|---|---|---|
| Illuminated shop name | <img src="assets/example_ler_shop_rgb.png" width="260"><br>RGB answer: **Hua Tian Hua** (wrong) | <img src="assets/example_ler_shop_measxyz.png" width="260"><br>PRSIMVL answer: **Zhengmei Dental Clinic** |
| Yellow sign text | <img src="assets/example_ler_sign_rgb.png" width="260"><br>RGB answer: **diamond** (wrong) | <img src="assets/example_ler_sign_measxyz.png" width="260"><br>PRSIMVL answer: **BLACK** |

Zoomed evidence crops:

| RGB crop | Meas.-XYZ crop | RGB crop | Meas.-XYZ crop |
|---|---|---|---|
| <img src="assets/example_ler_shop_crop_rgb.png" width="180"> | <img src="assets/example_ler_shop_crop_measxyz.png" width="180"> | <img src="assets/example_ler_sign_crop_rgb.png" width="180"> | <img src="assets/example_ler_sign_crop_measxyz.png" width="180"> |

## Earlier Synthetic-RAW Version

This release builds on our earlier synthetic-RAW prototype, [**End-to-End RAW Synergy for Elevated Vision-Language Reasoning**](https://openreview.net/forum?id=fsCtGojL2R), which introduced Raw-VLM with a learnable ISP frontend and RAW-tokenization for VLM reasoning. That first version used synthetic RAW data to study whether RAW sensor information can improve captioning, VQA, and hallucination behavior.

PRSIMVL extends that direction toward a release centered on measurement-grounded inputs: RAW-derived **Meas.-XYZ**, camera metadata grounding, MeasL-Bench-V1, MeasL-150K-V1, and released Qwen3-VL LoRA adapters.

## Main Results

The table reports the held-out MeasL-Bench protocol. BLEU and ROUGE-L are lexical metrics; LLM-Judge is reported as accuracy percentage.

| Model | Visual Input | BLEU | ROUGE-L | LLM-Judge |
|---|---|---:|---:|---:|
| Qwen3-VL-2B | RGB | 0.3407 | 0.3171 | 69.54 |
| Qwen3-VL-4B | RGB | 0.4442 | 0.3453 | 77.37 |
| Qwen3-VL-8B | RGB | 0.5046 | 0.3500 | 78.20 |
| **PRSIMVL-2B** | **Meas.-XYZ + metadata** | **0.5865** | **0.4244** | **77.99** |
| **PRSIMVL-4B** | **Meas.-XYZ + metadata** | **0.6021** | **0.4465** | **80.83** |
| **PRSIMVL-8B** | **Meas.-XYZ + metadata** | **0.6120** | **0.4571** | **82.66** |

<p align="center">
  <img src="assets/prsimvl_radar.png" alt="Capability radar" width="82%">
</p>

### Where It Helps Most

| Capability | RGB Qwen3-VL-8B BLEU / ROUGE-L | PRSIMVL-2B BLEU / ROUGE-L |
|---|---:|---:|
| HDR Evidence Recovery (HER) | 0.5343 / 0.3614 | **0.6066 / 0.4533** |
| Low-Illumination Evidence Recovery (LER) | 0.3470 / 0.2851 | **0.5174 / 0.4249** |
| Scene Text Recognition (STR) | 0.3719 / 0.3604 | **0.5084 / 0.4669** |
| General Visual Grounding (GVG) | 0.5109 / 0.3644 | **0.6117 / 0.4505** |
| Agent and Entity Identification (AEI) | 0.5304 / 0.4332 | **0.6210 / 0.5307** |
| Binary Visual Verification (BVV) | 0.5367 / 0.3580 | **0.6186 / 0.3732** |

## What Is In This Release

| Artifact | Location | Notes |
|---|---|---|
| Benchmark | [`eval_data/`](eval_data/) and [HF](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) | Matched Meas.-XYZ/RGB JSONL files and 3,812 images. |
| Training data | [`training_data/`](training_data/) and [HF](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) | 152,517 instruction-tuning examples and 48,000 images. |
| Demo inference | [`inference/`](inference/) | OpenAI-compatible `swift deploy` service demo for local images. |
| Evaluation wrapper | [`eval/`](eval/) | Reproducible MeasL-Bench inference and offline evaluation entrypoint. |
| Training configs | [`configs/qwen3_vl_150k_llmmeta_vit_proxy/`](configs/qwen3_vl_150k_llmmeta_vit_proxy/) | Launch scripts and SFT configs for 2B, 4B, and 8B. |
| Released adapters | [`exps/`](exps/) and [HF](https://huggingface.co/kepeng/PRSIMVL-LoRA-V1) | LoRA checkpoints for Qwen3-VL 2B, 4B, and 8B. |

## Benchmark And Data

### MeasL-Bench-V1

[MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) is the held-out benchmark for measurement-grounded language-vision evaluation.

| File | Rows / Files | Purpose |
|---|---:|---|
| `eval_data/test-raw-measl-bench.jsonl` | 2,183 rows | Main Meas.-XYZ benchmark. |
| `eval_data/test-rgb-measl-bench.jsonl` | 2,183 rows | Matched RGB benchmark. |
| `eval_data/image/` | 3,812 files | Local image assets referenced by both JSONL files. |

All JSONL image paths use the release-local form `eval_data/image/...`. When reading directly from the Hugging Face dataset repository root, remove the leading `eval_data/` prefix.

<details>
<summary>Capability taxonomy</summary>

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

</details>

### MeasL-150K-V1

[MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) is the released instruction-tuning corpus.

| File | Rows / Files | Purpose |
|---|---:|---|
| `training_data/train-measl-150k-v1.jsonl` | 152,517 rows | Final instruction-tuning set. |
| `training_data/image/` | 48,000 files | Release image subset referenced by the JSONL file. |

The corpus was built from approximately 700K auto-annotated candidates, filtered to 518,433 post-scoring records, balanced by source and question structure, and decontaminated against MeasL-Bench before release.

## Demo Inference

Start an OpenAI-compatible service with a released adapter:

```bash
conda activate msswiftv1_service
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapters exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000
```

Ask a question from another terminal:

```bash
conda activate msswiftv1_service
python inference/ask_service.py \
  --image inference/demo_data/images/demo1_pole_color.png \
  --question "This is a linear Image with Metadata: ISO: 250, Exposure Time: 1/640, Aperture: f/9. What is the color of the vertical pole visible through the windshield?"
```

See [`inference/README.md`](inference/README.md) for demo images, request options, and troubleshooting.

## Evaluation

Run the default Meas.-XYZ benchmark:

```bash
MODEL_SIZE=4b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

Run the matched RGB benchmark:

```bash
DATASET=rgb MODEL_SIZE=4b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

Enable LLM-as-judge through an OpenAI-compatible endpoint:

```bash
export JUDGE_API_KEY=YOUR_KEY
JUDGE_URL=https://openrouter.ai/api/v1 \
JUDGE_MODEL=openai/gpt-5 \
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

The evaluation entrypoint defaults to `eval_data/test-raw-measl-bench.jsonl` or `eval_data/test-rgb-measl-bench.jsonl`. Use `DATASET_FILE` and `IMAGE_ROOT` only for external datasets or non-standard image locations. Full options are documented in [`eval/README.md`](eval/README.md).

## Training

Final training configs are under [`configs/qwen3_vl_150k_llmmeta_vit_proxy/`](configs/qwen3_vl_150k_llmmeta_vit_proxy/).

```bash
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_2b.sh
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_4b.sh
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_8b.sh
```

The corresponding config files are:

- `sft_qwen3_vl_2b_prsimvl_v1.yaml`
- `sft_qwen3_vl_4b_prsimvl_v1.yaml`
- `sft_qwen3_vl_8b_prsimvl_v1.yaml`

## Released Weights

Released PRSIMVL LoRA weights are hosted on Hugging Face: [kepeng/PRSIMVL-LoRA-V1](https://huggingface.co/kepeng/PRSIMVL-LoRA-V1). The local release expects the same checkpoint layout under `exps/BANALCED_150K_META_VIT_PROXY/`.

| Size | Base Model | Local LoRA Checkpoint |
|---|---|---|
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000` |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-4B-Instruct/v12-20260425-113029/checkpoint-85000` |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-8B-Instruct/v2-20260423-205317/checkpoint-95000` |

## Repository Layout

```text
PRSIMVL/
├── assets/                       README figures and qualitative examples
├── eval/                         Benchmark inference and evaluation entrypoint
├── inference/                    Service-based VQA demo
├── eval_data/                    MeasL-Bench-V1 artifact folder
├── training_data/                MeasL-150K-V1 artifact folder
├── exps/                         Released LoRA adapters
├── configs/qwen3_vl_150k_llmmeta_vit_proxy/
│   └── PRSIMVL v1 training configs and launch scripts
├── scripts/test_raw_eval_pipeline_opt/
│   └── shared inference/evaluation implementation
└── swift/, libs/                 Training and inference code snapshot
```

## Release Notes

- Dataset license: CC BY-NC 4.0 for non-commercial research and education; citation is required.
- Evaluation outputs are written under `eval/output_benchmark/` by default.
- Generated scratch outputs, conversion utilities, and local environment checks are pruned from this public release snapshot.
- Contribution and release hygiene notes are available in [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

## Citation

Main paper: [Allegory of the Cave: Measurement-Grounded Vision-Language Learning](https://arxiv.org/abs/2605.11727)  
Earlier synthetic-RAW version: [End-to-End RAW Synergy for Elevated Vision-Language Reasoning](https://openreview.net/forum?id=fsCtGojL2R)  
Project page: <https://kepengxu.github.io/projects/prism-vl/>  
Author homepage: <https://kepengxu.github.io/>

```bibtex
@misc{xu2026allegory,
  title         = {Allegory of the Cave: Measurement-Grounded Vision-Language Learning},
  author        = {Xu, Kepeng and Xu, Li and He, Gang and Yu, Wenxin},
  year          = {2026},
  eprint        = {2605.11727},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.11727}
}

@inproceedings{xu2025rawvlm,
  title     = {End-to-End RAW Synergy for Elevated Vision-Language Reasoning},
  author    = {Xu, Kepeng and Qiao, Tong and Liu, Zhenyang and Xu, Li and He, Gang},
  booktitle = {IJCAI 2025 Workshop on Multimodal Knowledge and Language Modeling (MKLM)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=fsCtGojL2R}
}
```
