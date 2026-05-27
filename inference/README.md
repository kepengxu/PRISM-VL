# Service Inference Demo

This folder is the fastest way to try PRSIMVL on a local image. It starts a `swift deploy` service with a released adapter, then sends an OpenAI-compatible chat request containing one image and one question.

## Files

| File | Purpose |
|---|---|
| `ask_service.py` | Reads a local image, base64-encodes it, sends it to `/v1/chat/completions`, and prints the model answer. |
| `demo_data/images/` | Three small demo images for smoke testing the service. |
| `demo_data/demo_three.jsonl` | Three example requests in JSONL form. |
| `demo_data/demo_three_meta.json` | Human-readable metadata for the packaged demos. |

## Workflow

### 1. Enter the release environment

```bash
conda activate msswiftv1_service
cd /path/to/PRSIMVL
```

### 2. Start a model service

Pick an idle GPU and start `swift deploy` with a released adapter:

```bash
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --adapters exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000
```

The service listens on `http://127.0.0.1:8000` by default.

### 3. Ask a demo question

Run this in another terminal after the service is ready:

```bash
conda activate msswiftv1_service
cd /path/to/PRSIMVL

python inference/ask_service.py \
  --image inference/demo_data/images/demo1_pole_color.png \
  --question "This is a linear Image with Metadata: ISO: 250, Exposure Time: 1/640, Aperture: f/9. What is the color of the vertical pole visible through the windshield?"
```

## Demo Images

| Demo | Image | Question |
|---|---|---|
| 1 | `inference/demo_data/images/demo1_pole_color.png` | What is the color of the vertical pole visible through the windshield? |
| 2 | `inference/demo_data/images/demo2_shadow_check.png` | Are there sharp shadows on the walls that would indicate direct sunlight? |
| 3 | `inference/demo_data/images/demo3_orange_object.png` | What small orange object is placed next to the gray plush toy on the dashboard? |

## Custom Requests

```bash
python inference/ask_service.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen3-VL-2B-Instruct \
  --max-tokens 128 \
  --image /path/to/image.png \
  --question "This is a linear Image with Metadata: ISO: 800, Exposure Time: 1/60, Aperture: f/2.8. What evidence is visible?"
```

The helper intentionally prints only the model answer so it can be used in shell scripts or simple demos. Use the service logs for lower-level request debugging.

## Adapter Choices

Released LoRA weights are hosted on Hugging Face: [kepeng/PRSIMVL-LoRA-V1](https://huggingface.co/kepeng/PRSIMVL-LoRA-V1). Restore them under `exps/BANALCED_150K_META_VIT_PROXY/` before launching `swift deploy`.

| Size | Base model | Adapter path |
|---|---|---|
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/v8-20260421-133546/checkpoint-95000` |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-4B-Instruct/v12-20260425-113029/checkpoint-85000` |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | `exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-8B-Instruct/v2-20260423-205317/checkpoint-95000` |

## Troubleshooting

| Symptom | Check |
|---|---|
| `Connection refused` | The `swift deploy` service is not ready or is listening on another host/port. |
| `Image not found` | The `--image` path is resolved from the current working directory; run from the release root or pass an absolute path. |
| Very long answer | Lower `--max-tokens`. |
| Poor answer quality | Include capture metadata in the question, matching the training/evaluation prompt style. |

## Related Resources

- Main release README: [`../README.md`](../README.md)
- Evaluation workflow: [`../eval/README.md`](../eval/README.md)
- Project page: <https://kepengxu.github.io/projects/prism-vl/>
- arXiv: <https://arxiv.org/abs/2605.11727>
