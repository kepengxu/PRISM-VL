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
- instruction-tuning
- measurement-grounding
- raw-image
- camera-raw
- meas-xyz
- qwen3-vl
- prsimvl
pretty_name: MeasL-150K-V1
size_categories:
- 100K<n<1M
---

# MeasL-150K-V1

**MeasL-150K-V1** is the official PRSIMVL instruction-tuning corpus for measurement-grounded visual reasoning. It is aligned with the [MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) evaluation protocol and released for reproducible non-commercial research.

- **Hugging Face**: [kepeng/MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1)
- **Project page**: <https://kepengxu.github.io/projects/prism-vl/>
- **Paper**: <https://arxiv.org/abs/2605.11727>

## At A Glance

| Item | Value |
|---|---|
| Samples | 152,517 instruction-tuning examples. |
| Images | 48,000 release image files. |
| Primary use | Multimodal SFT for Meas.-XYZ and metadata-grounded reasoning. |
| Benchmark relation | Decontaminated against MeasL-Bench-V1. |
| License | CC BY-NC 4.0, non-commercial research and education. |

## Release Contents

| File | Scale | Description |
|---|---:|---|
| `train-measl-150k-v1.jsonl` | 152,517 samples | Curated instruction-tuning corpus. |
| `image/` | 48,000 files | Release image subset referenced by `train-measl-150k-v1.jsonl`. |

## Path Convention

Training records use local relative image paths under:

```text
training_data/image/
```

When this folder is placed under the release root, the training configs can resolve images directly from the packaged layout.

## Dataset Construction Summary

The released corpus was built from approximately 700K auto-annotated candidates. After quality scoring, 518,433 records remained. The final 152,517 examples were balanced by source and question structure, packaged with release-local images, and checked to remove strict benchmark overlaps.

## JSONL Format

Each row follows the multimodal conversation format used by the release training configs. Typical fields include:

| Field | Meaning |
|---|---|
| `messages` | Conversation-style prompt and target response. |
| `images` | One or more paths under `training_data/image/`. |
| metadata fields | Capture/context signals used for measurement grounding. |

## Dataset Registration

The release registry maps these aliases to the release-local training JSONL:

- `MEASL/TRAIN_150K_V1`
- `MEASL/TRAIN_150K_V1_EXTRA`

Training launch scripts are documented in the main release README and live under:

```text
configs/qwen3_vl_150k_llmmeta_vit_proxy/
```

## Quality And Integrity

- Benchmark decontamination is applied before release.
- The corpus is intended for reproducible instruction tuning, not benchmark evaluation.
- Modified or expanded variants should be published under a new semantic version, for example `MeasL-150K-V2`.

## License

This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You may use, share, and adapt the dataset for non-commercial research and educational purposes, provided that you give appropriate credit and cite the associated project or paper. Commercial use is not permitted without prior written permission from the authors.
