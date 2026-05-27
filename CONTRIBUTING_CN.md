# 贡献指南

这是一个 metadata-aware Qwen3-VL proxy benchmark 的发布仓库。贡献应优先保证可复现、可运行、易检查。

适合提交的改动：

- 修复路径、文档或安装说明。
- 改进 benchmark runner，但不要静默改变默认指标语义。
- 为 release 脚本或数据校验补充聚焦测试。
- 报告缺失的 Git LFS 资源或损坏的 benchmark 文件。

提交前建议运行：

```bash
python -m py_compile eval/benchmark_pipeline.py eval/infer_and_eval.py inference/demo_infer.py scripts/test_raw_eval_pipeline_opt/pipeline.py
bash -n eval/run_infer_and_eval.sh
bash -n inference/run_demo_inference.sh
bash eval/run_infer_and_eval.sh --dry-run
bash inference/run_demo_inference.sh --dry-run
```

不要提交评估输出、optimizer state、本地日志或缓存目录。
