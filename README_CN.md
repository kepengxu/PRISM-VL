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
  <a href="https://kepengxu.github.io/projects/prism-vl/">项目主页</a> ·
  <a href="https://arxiv.org/abs/2605.11727">arXiv:2605.11727</a> ·
  <a href="https://huggingface.co/datasets/kepeng/MeasL-Bench-V1">MeasL-Bench-V1</a> ·
  <a href="https://huggingface.co/datasets/kepeng/MeasL-150K-V1">MeasL-150K-V1</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <img alt="Method" src="https://img.shields.io/badge/Method-PRSIMVL-111827">
  <img alt="Input" src="https://img.shields.io/badge/Input-Meas.--XYZ-0a7ea4">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-MeasL--Bench--V1-2ea44f">
  <img alt="Training" src="https://img.shields.io/badge/Training-MeasL--150K--V1-f0883e">
  <img alt="Backbone" src="https://img.shields.io/badge/Base-Qwen3--VL-1f6feb">
  <img alt="Adapters" src="https://img.shields.io/badge/LoRA-2B%20%7C%204B%20%7C%208B-8957e5">
</p>

**PRSIMVL** 是一个面向测量域视觉语言学习的研究发布版本。它关注一个直接但重要的问题：当 RGB 渲染已经丢失传感器证据时，VLM 是否可以从测量域观察中获得更可靠的推理能力？

PRSIMVL 保留 Qwen3-VL 的训练、推理和评估流程，但将视觉观察接口从 ISP 后的 RGB 前移到 RAW 派生的 **Meas.-XYZ**，并结合 ISO、曝光时间、光圈等相机元数据进行 grounding。本 release 包含 benchmark、训练集、评估 pipeline、service demo 和 LoRA checkpoints。

<p align="center">
  <img src="assets/prsimvl_framework.png" alt="PRSIMVL framework" width="92%">
</p>

## 30 秒摘要

| 项目 | 内容 |
|---|---|
| 核心思想 | 使用 RAW 派生的 Meas.-XYZ 和拍摄元数据，缓解 RGB 渲染裁剪、降噪、tone mapping、量化带来的证据损失。 |
| Benchmark | [MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1)，2,183 条 held-out matched examples，覆盖 14 个测量敏感能力切片。 |
| 训练集 | [MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1)，152,517 条 instruction-tuning 样本，引用 48,000 个 release images。 |
| 模型 | Qwen3-VL 2B、4B、8B，对应 PRSIMVL LoRA adapters。 |
| 主要结果 | PRSIMVL-8B 相比 RGB Qwen3-VL-8B 在 MeasL-Bench 上提升 **+0.1074 BLEU**、**+0.1071 ROUGE-L** 和 **+4.46 LLM-Judge points**。 |

## 从这里开始

| 目标 | 入口 | 说明 |
|---|---|---|
| 问一张图 | [`inference/README.md`](inference/README.md) | 启动 `swift deploy`，用 `ask_service.py` 发送本地图像和问题。 |
| 跑 benchmark | [`eval/README.md`](eval/README.md) | 评估 Meas.-XYZ 或 matched RGB split。 |
| 看评估数据 | [`eval_data/README.md`](eval_data/README.md) | Dataset card、taxonomy、schema、HF loading snippet、路径规则。 |
| 看训练数据 | [`training_data/README.md`](training_data/README.md) | Dataset card、发布内容、质量控制、registry aliases。 |
| 看 release 范围 | [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md) | 说明 release 中包含、裁剪和需要恢复的大文件。 |

## 快速开始

安装 release snapshot：

```bash
git clone <repo-url> PRSIMVL
cd PRSIMVL
bash install_editable.sh
```

先 dry-run，检查命令而不启动模型推理：

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh --dry-run
```

运行一个 PRSIMVL adapter 的默认 Meas.-XYZ benchmark：

```bash
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

大体积 artifacts 需要放在 release-local 路径：

```text
eval_data/       # MeasL-Bench-V1 JSONL + image/
training_data/   # MeasL-150K-V1 JSONL + image/
exps/            # released LoRA adapters
```

公开数据集位于 Hugging Face：[MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) 用于评估，[MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) 用于训练。完整 adapter 推理或评估前，需要将 released adapters 恢复到 `exps/`。

## 为什么需要测量域 Grounding

RGB 是面向显示的人类可视化结果，不是相机最初记录的物理测量。ISP 过程中的裁剪、降噪、tone mapping 和量化可能会在模型看到图像前移除关键证据。PRSIMVL 将相机测量作为一等观察对象：Meas.-XYZ 保留 RAW 派生的线性三通道信息，元数据提供拍摄上下文。

下面的低照文本案例展示了 RGB 与 Meas.-XYZ 的差异：RGB 可能暴露错误证据，而 Meas.-XYZ 保留了可恢复的答案区域。

| Case | RGB Observation | Meas.-XYZ Observation |
|---|---|---|
| Illuminated shop name | <img src="assets/example_ler_shop_rgb.png" width="260"><br>RGB answer: **Hua Tian Hua** (wrong) | <img src="assets/example_ler_shop_measxyz.png" width="260"><br>PRSIMVL answer: **Zhengmei Dental Clinic** |
| Yellow sign text | <img src="assets/example_ler_sign_rgb.png" width="260"><br>RGB answer: **diamond** (wrong) | <img src="assets/example_ler_sign_measxyz.png" width="260"><br>PRSIMVL answer: **BLACK** |

局部证据 crop：

| RGB crop | Meas.-XYZ crop | RGB crop | Meas.-XYZ crop |
|---|---|---|---|
| <img src="assets/example_ler_shop_crop_rgb.png" width="180"> | <img src="assets/example_ler_shop_crop_measxyz.png" width="180"> | <img src="assets/example_ler_sign_crop_rgb.png" width="180"> | <img src="assets/example_ler_sign_crop_measxyz.png" width="180"> |

## 主结果

下表汇报 held-out MeasL-Bench protocol。BLEU 和 ROUGE-L 是文本指标，LLM-Judge 以准确率百分比汇报。

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

### 主要提升场景

| Capability | RGB Qwen3-VL-8B BLEU / ROUGE-L | PRSIMVL-2B BLEU / ROUGE-L |
|---|---:|---:|
| HDR Evidence Recovery (HER) | 0.5343 / 0.3614 | **0.6066 / 0.4533** |
| Low-Illumination Evidence Recovery (LER) | 0.3470 / 0.2851 | **0.5174 / 0.4249** |
| Scene Text Recognition (STR) | 0.3719 / 0.3604 | **0.5084 / 0.4669** |
| General Visual Grounding (GVG) | 0.5109 / 0.3644 | **0.6117 / 0.4505** |
| Agent and Entity Identification (AEI) | 0.5304 / 0.4332 | **0.6210 / 0.5307** |
| Binary Visual Verification (BVV) | 0.5367 / 0.3580 | **0.6186 / 0.3732** |

## Release 内容

| Artifact | 位置 | 说明 |
|---|---|---|
| Benchmark | [`eval_data/`](eval_data/) 和 [HF](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) | Matched Meas.-XYZ/RGB JSONL files 和 3,812 张图像。 |
| 训练数据 | [`training_data/`](training_data/) 和 [HF](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) | 152,517 条 instruction-tuning 样本和 48,000 张图像。 |
| Demo 推理 | [`inference/`](inference/) | 面向本地图像的 OpenAI-compatible `swift deploy` service demo。 |
| 评估入口 | [`eval/`](eval/) | 可复现的 MeasL-Bench 推理和离线评估入口。 |
| 训练配置 | [`configs/qwen3_vl_150k_llmmeta_vit_proxy/`](configs/qwen3_vl_150k_llmmeta_vit_proxy/) | 2B、4B、8B 的 launch scripts 和 SFT configs。 |
| Released adapters | `exps/` | LoRA checkpoints，路径见下表。 |

## 数据集

### MeasL-Bench-V1

[MeasL-Bench-V1](https://huggingface.co/datasets/kepeng/MeasL-Bench-V1) 是面向测量域语言视觉评估的 held-out benchmark。

| File | Rows / Files | Purpose |
|---|---:|---|
| `eval_data/test-raw-measl-bench.jsonl` | 2,183 rows | Main Meas.-XYZ benchmark。 |
| `eval_data/test-rgb-measl-bench.jsonl` | 2,183 rows | Matched RGB benchmark。 |
| `eval_data/image/` | 3,812 files | 两个 JSONL 共同引用的图像文件。 |

JSONL 图片路径使用 `eval_data/image/...`。如果直接从 Hugging Face dataset repository root 读取，需要去掉前缀 `eval_data/`。

### MeasL-150K-V1

[MeasL-150K-V1](https://huggingface.co/datasets/kepeng/MeasL-150K-V1) 是发布版 instruction-tuning corpus。

| File | Rows / Files | Purpose |
|---|---:|---|
| `training_data/train-measl-150k-v1.jsonl` | 152,517 rows | Final instruction-tuning set。 |
| `training_data/image/` | 48,000 files | JSONL 引用的 release image subset。 |

训练集从约 700K 条自动标注候选构建，经打分过滤得到 518,433 条记录，再按来源和问题结构平衡，最终发布 152,517 条样本。发布前已去除与 MeasL-Bench 的严格 overlap。

## Demo 推理

启动带 released adapter 的 OpenAI-compatible service：

```bash
conda activate msswiftv1_service
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapters exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000
```

另开终端提问：

```bash
conda activate msswiftv1_service
python inference/ask_service.py \
  --image inference/demo_data/images/demo1_pole_color.png \
  --question "This is a linear Image with Metadata: ISO: 250, Exposure Time: 1/640, Aperture: f/9. What is the color of the vertical pole visible through the windshield?"
```

更多 demo images、请求参数和排错说明见 [`inference/README.md`](inference/README.md)。

## 评估

运行默认 Meas.-XYZ benchmark：

```bash
MODEL_SIZE=4b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

运行 matched RGB benchmark：

```bash
DATASET=rgb MODEL_SIZE=4b CUDA_VISIBLE_DEVICES=0 bash eval/run_infer_and_eval.sh
```

通过 OpenAI-compatible endpoint 开启 LLM-as-judge：

```bash
export JUDGE_API_KEY=YOUR_KEY
JUDGE_URL=https://openrouter.ai/api/v1 \
JUDGE_MODEL=openai/gpt-5 \
MODEL_SIZE=2b CUDA_VISIBLE_DEVICES=0 \
bash eval/run_infer_and_eval.sh
```

评估入口默认使用 `eval_data/test-raw-measl-bench.jsonl` 或 `eval_data/test-rgb-measl-bench.jsonl`。外部数据集或非标准图片路径才需要设置 `DATASET_FILE` 和 `IMAGE_ROOT`。完整说明见 [`eval/README.md`](eval/README.md)。

## 训练

最终训练配置位于 [`configs/qwen3_vl_150k_llmmeta_vit_proxy/`](configs/qwen3_vl_150k_llmmeta_vit_proxy/)。

```bash
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_2b.sh
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_4b.sh
bash configs/qwen3_vl_150k_llmmeta_vit_proxy/train_prsimvl_8b.sh
```

对应配置文件：

- `sft_qwen3_vl_2b_prsimvl_v1.yaml`
- `sft_qwen3_vl_4b_prsimvl_v1.yaml`
- `sft_qwen3_vl_8b_prsimvl_v1.yaml`

## 发布权重

| Size | Base Model | LoRA Checkpoint |
|---|---|---|
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000` |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-4B-Instruct/v12-20260425-113029/checkpoint-85000` |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-8B-Instruct/v2-20260423-205317/checkpoint-95000` |

## 项目结构

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

## Release 说明

- 数据集 license：CC BY-NC 4.0，仅允许非商业研究和教育使用，使用时需要引用。
- 评估输出默认写入 `eval/output_benchmark/`。
- 生成输出、转换工具和本地环境检查脚本已从 public release snapshot 中裁剪。
- 贡献和发布规范见 [`CONTRIBUTING.md`](CONTRIBUTING.md)、[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)、[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)。

## 引用

论文: [Allegory of the Cave: Measurement-Grounded Vision-Language Learning](https://arxiv.org/abs/2605.11727)  
项目主页: <https://kepengxu.github.io/projects/prism-vl/>  
作者主页: <https://kepengxu.github.io/>

```bibtex
@misc{xu2026allegory,
  title         = {Allegory of the Cave: Measurement-Grounded Vision-Language Learning},
  author        = {Xu, Kepeng and Xu, Li and He, Gang and Yu, Wenxin},
  year          = {2026},
  eprint        = {2605.11727},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.11727}
}
```
