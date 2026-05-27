from __future__ import annotations

import copy
import json
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BENCHMARK_CATEGORIES = (
    'CAG',
    'NG',
    'DSG',
    'HER',
    'LER',
    'STR',
    'GVG',
    'CVR',
    'SRU',
    'MSQ',
    'EAQ',
    'DS',
    'AEI',
    'BVV',
)

OLD_OR_REGROUP_CATEGORIES = {
    'color',
    'counting',
    'description',
    'ex_OCR',
    'ex_hdr',
    'ex_lowlight',
    'hdr',
    'lowlight',
    'ocr_text',
    'other',
    'reasoning',
    'spatial',
    'wh_how',
    'wh_what',
    'wh_which',
    'wh_who',
    'yes_no',
}


@dataclass(frozen=True)
class ModelSpec:
    model: str
    adapter: str
    default_batch_size: int


MODEL_SPECS = {
    '2b': ModelSpec(
        model='Qwen/Qwen3-VL-2B-Instruct',
        adapter='exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-2B-Instruct/'
        'v8-20260421-133546/checkpoint-95000',
        default_batch_size=16,
    ),
    '4b': ModelSpec(
        model='Qwen/Qwen3-VL-4B-Instruct',
        adapter='exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-4B-Instruct/'
        'v12-20260425-113029/checkpoint-85000',
        default_batch_size=24,
    ),
    '8b': ModelSpec(
        model='Qwen/Qwen3-VL-8B-Instruct',
        adapter='exps/BANALCED_150K_META_VIT_PROXY/output-Qwen3-VL-8B-Instruct/'
        'v2-20260423-205317/checkpoint-95000',
        default_batch_size=12,
    ),
}


@dataclass(frozen=True)
class PreparedDataset:
    source_file: Path
    dataset_file: Path
    rows: int
    counts: Counter


def release_root(current_file: str | Path) -> Path:
    return Path(current_file).resolve().parents[1]


def resolve_path(root: Path, path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return root / path


def command_path(root: Path, path: str | Path) -> str:
    resolved = resolve_path(root, path)
    try:
        return str(resolved.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def shell_join(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def shell_join_redacted(command: Iterable[str], secret_flags: set[str] | None = None) -> str:
    secret_flags = secret_flags or {'--judge-api-key'}
    redacted = []
    hide_next = False
    for part in command:
        part = str(part)
        if hide_next:
            redacted.append('<redacted>')
            hide_next = False
            continue
        redacted.append(part)
        if part in secret_flags:
            hide_next = True
    return shell_join(redacted)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_no}: {exc}') from exc
            if not isinstance(item, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            items.append(item)
    return items


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def validate_benchmark_categories(
    items: list[dict],
    dataset_file: Path,
    require_all_categories: bool = True,
) -> Counter:
    counts = Counter(str(item.get('question_type', 'unknown')) for item in items)
    allowed = set(BENCHMARK_CATEGORIES)
    unknown = sorted(set(counts) - allowed)
    if unknown:
        old = sorted(set(unknown) & OLD_OR_REGROUP_CATEGORIES)
        details = f'old/regroup labels still present: {old}' if old else f'unexpected labels: {unknown}'
        raise ValueError(
            f'{dataset_file} is not benchmark-renamed. Found {unknown}; {details}. '
            f'Expected only: {", ".join(BENCHMARK_CATEGORIES)}')

    if require_all_categories:
        missing = [category for category in BENCHMARK_CATEGORIES if counts.get(category, 0) == 0]
        if missing:
            raise ValueError(f'{dataset_file} is missing benchmark categories: {missing}')
    return counts


def format_counts(counts: Counter) -> str:
    return ', '.join(f'{category}={counts.get(category, 0)}' for category in BENCHMARK_CATEGORIES)


def select_samples_by_category(items: list[dict], samples_per_category: int) -> list[dict]:
    if samples_per_category <= 0:
        raise ValueError('--samples-per-category must be positive')

    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[str(item.get('question_type', 'unknown'))].append(item)

    missing = [
        category for category in BENCHMARK_CATEGORIES
        if len(buckets.get(category, [])) < samples_per_category
    ]
    if missing:
        raise ValueError(
            f'Not enough samples for categories {missing}; requested {samples_per_category} each')

    selected = []
    for category in BENCHMARK_CATEGORIES:
        selected.extend(buckets[category][:samples_per_category])
    return selected


def _get_image_path(image) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return str(image.get('path', ''))
    return ''


def _set_image_path(image, path: str):
    if isinstance(image, str):
        return path
    if isinstance(image, dict):
        updated = dict(image)
        updated['path'] = path
        return updated
    return image


def rewrite_image_paths(items: list[dict], image_root: Path) -> list[dict]:
    rewritten = []
    for item in items:
        updated = copy.deepcopy(item)
        images = updated.get('images') or []
        new_images = []
        for image in images:
            image_path = _get_image_path(image)
            if image_path:
                path = Path(image_path).expanduser()
                if not path.is_absolute():
                    path_parts = path.parts
                    if len(path_parts) >= 3 and path_parts[0] == 'eval_data' and path_parts[1] in {'image', 'images'}:
                        path = Path(*path_parts[2:])
                    path = image_root / path
                image = _set_image_path(image, str(path))
            new_images.append(image)
        updated['images'] = new_images
        rewritten.append(updated)
    return rewritten


def collect_missing_images(items: list[dict], root: Path, max_examples: int = 10) -> list[str]:
    missing = []
    seen = set()
    for item in items:
        for image in item.get('images') or []:
            image_path = _get_image_path(image)
            if not image_path:
                continue
            path = Path(image_path).expanduser()
            if not path.is_absolute():
                path = root / path
            if path.exists():
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            missing.append(key)
            if len(missing) >= max_examples:
                return missing
    return missing


def prepare_dataset(
    root: Path,
    dataset_file: str | Path,
    output_file: str | Path | None = None,
    image_root: str | Path | None = None,
    samples_per_category: int | None = None,
    require_all_categories: bool = True,
) -> PreparedDataset:
    source_file = resolve_path(root, dataset_file)
    items = load_jsonl(source_file)
    validate_benchmark_categories(items, source_file, require_all_categories=require_all_categories)

    if samples_per_category is not None:
        items = select_samples_by_category(items, samples_per_category)

    if image_root:
        image_root_path = resolve_path(root, image_root)
        items = rewrite_image_paths(items, image_root_path)

    counts = validate_benchmark_categories(
        items,
        source_file,
        require_all_categories=require_all_categories,
    )

    if output_file or image_root or samples_per_category is not None:
        if output_file is None:
            raise ValueError('output_file is required when rewriting or sampling the dataset')
        prepared_file = resolve_path(root, output_file)
        write_jsonl(prepared_file, items)
    else:
        prepared_file = source_file

    return PreparedDataset(
        source_file=source_file,
        dataset_file=prepared_file,
        rows=len(items),
        counts=counts,
    )


def build_common_pipeline_args(
    *,
    swift_bin: str,
    model: str,
    adapter: str,
    dataset_file: str,
    output_root: str,
    max_new_tokens: int,
    max_batch_size: int,
    infer_backend: str,
    cuda_visible_devices: str | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    vllm_max_model_len: int | None = None,
) -> list[str]:
    args = [
        '--swift-bin',
        swift_bin,
        '--model',
        model,
        '--adapter',
        adapter,
        '--dataset-file',
        dataset_file,
        '--output-root',
        output_root,
        '--max-new-tokens',
        str(max_new_tokens),
        '--max-batch-size',
        str(max_batch_size),
        '--infer-backend',
        infer_backend,
    ]
    if cuda_visible_devices:
        args.extend(['--cuda-visible-devices', cuda_visible_devices])
    if infer_backend == 'vllm':
        if vllm_gpu_memory_utilization is not None:
            args.extend(['--vllm-gpu-memory-utilization', str(vllm_gpu_memory_utilization)])
        if vllm_max_model_len is not None:
            args.extend(['--vllm-max-model-len', str(vllm_max_model_len)])
    return args
