from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def sanitize_name(value: str) -> str:
    value = value.strip().replace('\\', '/')
    value = value.replace('/', '_')
    value = re.sub(r'[^0-9A-Za-z._+-]+', '_', value)
    return value or 'unknown'


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    run_name: str

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def pred_file(self) -> Path:
        return self.run_dir / 'test_raw_predictions.jsonl'

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / 'eval_results_offline'

    @property
    def run_config_file(self) -> Path:
        return self.run_dir / 'run_config.json'

    @property
    def latest_file(self) -> Path:
        return self.output_root / 'latest_run.json'


def build_run_name(model: str, adapter: str, dataset_file: str) -> str:
    dataset_name = Path(dataset_file).stem
    return '+'.join([
        sanitize_name(model),
        sanitize_name(adapter or 'None'),
        sanitize_name(dataset_name),
    ])


def save_latest_run(paths: RunPaths) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_name': paths.run_name,
        'run_dir': str(paths.run_dir),
        'pred_file': str(paths.pred_file),
        'eval_dir': str(paths.eval_dir),
    }
    with open(paths.latest_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_latest_run(output_root: Path) -> dict | None:
    latest_file = output_root / 'latest_run.json'
    if not latest_file.exists():
        return None
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)
