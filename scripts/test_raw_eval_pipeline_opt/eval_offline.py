#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评测脚本：对单个 swift infer JSONL 推理结果进行多指标评测 + LLM-as-Judge

输入:
  --pred_file    单个推理结果文件（JSONL）
  --dataset_file 单个数据文件（JSON / JSONL，包含 ground truth 与 question_type）

评测指标:
  - 精确匹配 / 包含匹配 / 模糊匹配
  - BLEU-4 / ROUGE-L
  - LLM-as-Judge（精确匹配通过的跳过 Judge，节省开销）
  - 按 question_type 分组统计
"""

import re
import json
import argparse
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


FUZZY_THRESHOLD = 0.8

_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
_smoothie = SmoothingFunction().method1


def parse_args():
    p = argparse.ArgumentParser(description='离线评测脚本（单文件，多指标 + LLM Judge）')
    p.add_argument('--pred_file', required=True, help='单个推理结果 JSONL 文件')
    p.add_argument('--dataset_file', required=True, help='单个数据文件（JSON / JSONL）')
    p.add_argument('--output_dir', default=None, help='输出目录（默认: pred_file 同级 eval_results_offline/）')
    p.add_argument('--fuzzy_threshold', type=float, default=FUZZY_THRESHOLD, help='模糊匹配阈值')

    # LLM Judge
    p.add_argument('--judge_url', default=None, help='Judge API base URL')
    p.add_argument('--judge_model', default=None, help='Judge 模型名（默认自动检测）')
    p.add_argument('--judge_api_key', default=None, help='Judge API Key')
    p.add_argument('--judge_max_workers', type=int, default=8, help='Judge 并发数')
    p.add_argument('--judge_max_tokens', type=int, default=10240, help='Judge 最大生成 token')
    p.add_argument('--judge_max_retries', type=int, default=5, help='Judge 最大重试次数')

    # Resume
    p.add_argument('--resume', default=None, help='之前的结果 JSON 文件，跳过已有 judge 结果')

    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def multi_level_match(prediction: str, ground_truth: str, threshold: float) -> Dict[str, Any]:
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)
    exact = (pred == gt)
    contain = (gt in pred) if gt else False
    fuzzy_score = SequenceMatcher(None, pred, gt).ratio()
    fuzzy = (fuzzy_score >= threshold)
    return {
        'exact': exact,
        'contain': contain,
        'fuzzy': fuzzy,
        'fuzzy_score': round(fuzzy_score, 4),
        'correct_any': exact or contain or fuzzy,
    }


def compute_bleu_rouge(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = list(prediction.strip().lower())
    gt_tokens = list(ground_truth.strip().lower())
    try:
        bleu = sentence_bleu(
            [gt_tokens], pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_smoothie)
    except Exception:
        bleu = 0.0
    try:
        rl = _rouge.score(ground_truth.strip(), prediction.strip())['rougeL'].fmeasure
    except Exception:
        rl = 0.0
    return {'bleu': round(bleu, 4), 'rouge_l': round(rl, 4)}


def extract_question_type(item: dict) -> str:
    question_type = item.get('question_type', 'unknown')
    return str(question_type) if question_type is not None else 'unknown'


def make_key(item: dict) -> str:
    query = ''
    for msg in item.get('messages', []):
        if msg.get('role') == 'user':
            query = msg.get('content', '')
            break
    images = item.get('images', [])
    if images:
        img = images[0]
        if isinstance(img, str):
            img_path = img
        elif isinstance(img, dict):
            img_path = img.get('path', '')
        else:
            img_path = ''
    else:
        img_path = ''
    return f'{img_path}|||{query}|||{extract_question_type(item)}'


def load_json_or_jsonl(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix == '.jsonl':
            items = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return items
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f'Unsupported JSON structure in {path}, expected a JSON array.')


def extract_question(content: str) -> str:
    q = content.replace('<image>', '').strip()
    q = q.replace('\\n', '\n').strip()
    return q


def match_pred_to_gt(dataset_items: list[dict], pred_items: list[dict]):
    pred_by_key = defaultdict(deque)
    for item in pred_items:
        pred_by_key[make_key(item)].append(item)

    matched = []
    unmatched_gt = []
    for gt_item in dataset_items:
        key = make_key(gt_item)
        pred_bucket = pred_by_key.get(key)
        if pred_bucket:
            pred_item = pred_bucket.popleft()
            matched.append((gt_item, pred_item))
            if not pred_bucket:
                pred_by_key.pop(key, None)
        else:
            unmatched_gt.append(gt_item)

    unmatched_pred = [item for pred_bucket in pred_by_key.values() for item in pred_bucket]
    return matched, unmatched_gt, unmatched_pred


def make_resume_key(image_path: str, question: str, ground_truth: str, question_type: str) -> str:
    return f'{image_path}|||{question}|||{ground_truth}|||{question_type}'


def detect_judge_model(judge_url: str, api_key: Optional[str] = None) -> Optional[str]:
    try:
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else None
        resp = requests.get(f'{judge_url}/models', headers=headers, timeout=30)
        if resp.status_code == 200:
            models = resp.json().get('data', [])
            if models:
                return models[0].get('id')
    except Exception:
        pass
    return None


def call_llm_judge(
    judge_url: str, judge_model: str,
    question: str, ground_truth: str, prediction: str,
    api_key: Optional[str] = None, max_tokens: int = 10240,
) -> Dict[str, Any]:
    prompt = (
        "你是一个严格的评测裁判。请判断【模型回答】与【标准答案】的语义是否一致。\n"
        "只需回答一个 JSON: {\"correct\": true} 或 {\"correct\": false}，不要输出其他内容。\n\n"
        f"【问题】{question}\n"
        f"【标准答案】{ground_truth}\n"
        f"【模型回答】{prediction}\n"
    )
    try:
        payload = {
            'model': judge_model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else None
        print(f'[Judge Request] model={judge_model} question={question[:120]!r}')
        print(f'{judge_url}/chat/completions')
        resp = requests.post(f'{judge_url}/chat/completions',
                             json=payload, headers=headers, timeout=60)
        print(f'[Judge Response] status_code={resp.status_code}')
        if resp.status_code != 200:
            print(f'[Judge Response Body] {resp.text[:2000]}')
            return {'judge_correct': None, 'judge_reason': None, 'judge_error': f'HTTP {resp.status_code}'}

        response_json = resp.json()
        message = ((response_json.get('choices') or [{}])[0].get('message') or {})
        content = message.get('content')
        if isinstance(content, list):
            text_parts = [part.get('text', '') for part in content if isinstance(part, dict)]
            content = ''.join(text_parts)
        if content is None:
            content = message.get('reasoning_content')
        if content is None and message.get('tool_calls'):
            content = json.dumps(message['tool_calls'], ensure_ascii=False)
        if content is None:
            print(f'[Judge Response Body] {json.dumps(response_json, ensure_ascii=False)[:2000]}')
            return {'judge_correct': None, 'judge_reason': None, 'judge_error': 'Judge content is empty'}
        content = str(content).strip()
        print(f'[Judge Parsed Content] {content[:2000]}')
        low = content.lower()
        if '"correct": true' in low or '"correct":true' in low:
            return {'judge_correct': True, 'judge_reason': content, 'judge_error': None}
        if '"correct": false' in low or '"correct":false' in low:
            return {'judge_correct': False, 'judge_reason': content, 'judge_error': None}
        if 'true' in low and 'false' not in low:
            return {'judge_correct': True, 'judge_reason': content, 'judge_error': None}
        if 'false' in low:
            return {'judge_correct': False, 'judge_reason': content, 'judge_error': None}
        if any(w in content for w in ('一致', '正确', '相同', '相符', '匹配')) \
                and not any(w in content for w in ('不一致', '不正确', '不相同', '不相符', '不匹配')):
            return {'judge_correct': True, 'judge_reason': content, 'judge_error': None}
        if any(w in content for w in ('不一致', '不正确', '不相同', '不相符', '不匹配', '错误', '不同')):
            return {'judge_correct': False, 'judge_reason': content, 'judge_error': None}
        return {'judge_correct': None, 'judge_reason': content, 'judge_error': '无法解析 Judge 回复'}
    except Exception as e:
        print(f'[Judge Exception] {type(e).__name__}: {e}')
        return {'judge_correct': None, 'judge_reason': None, 'judge_error': str(e)}


def init_bucket(question_type: str) -> Dict[str, Any]:
    return {
        'question_type': question_type,
        'total': 0,
        'matched': 0,
        'missing_pred': 0,
        'unmatched_pred': 0,
        'exact_correct': 0,
        'contain_correct': 0,
        'fuzzy_correct': 0,
        'any_correct': 0,
        'sum_bleu': 0.0,
        'sum_rouge_l': 0.0,
        'judge_correct': 0,
        'judge_total': 0,
    }


def finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    total = bucket['total']
    judge_total = bucket['judge_total']
    bucket['exact_accuracy'] = bucket['exact_correct'] / total if total else 0
    bucket['contain_accuracy'] = bucket['contain_correct'] / total if total else 0
    bucket['fuzzy_accuracy'] = bucket['fuzzy_correct'] / total if total else 0
    bucket['any_accuracy'] = bucket['any_correct'] / total if total else 0
    bucket['avg_bleu'] = bucket['sum_bleu'] / total if total else 0
    bucket['avg_rouge_l'] = bucket['sum_rouge_l'] / total if total else 0
    bucket['judge_accuracy'] = bucket['judge_correct'] / judge_total if judge_total else 0
    bucket.pop('sum_bleu', None)
    bucket.pop('sum_rouge_l', None)
    return bucket


def summarize_by_question_type(results: list[dict], unmatched_pred_count: int) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in results:
        question_type = r.get('question_type', 'unknown') or 'unknown'
        bucket = buckets.setdefault(question_type, init_bucket(question_type))
        bucket['total'] += 1
        if r.get('prediction') is not None:
            bucket['matched'] += 1
        else:
            bucket['missing_pred'] += 1
        match = r.get('match', {})
        bucket['exact_correct'] += int(match.get('exact', False))
        bucket['contain_correct'] += int(match.get('contain', False))
        bucket['fuzzy_correct'] += int(match.get('fuzzy', False))
        bucket['any_correct'] += int(match.get('correct_any', False))
        bucket['sum_bleu'] += r.get('bleu', 0.0)
        bucket['sum_rouge_l'] += r.get('rouge_l', 0.0)
        if r.get('judge_correct') is not None:
            bucket['judge_total'] += 1
            bucket['judge_correct'] += int(r.get('judge_correct') is True)

    if unmatched_pred_count:
        buckets.setdefault('_unmatched_predictions', init_bucket('_unmatched_predictions'))['unmatched_pred'] = unmatched_pred_count

    return {k: finalize_bucket(v) for k, v in sorted(buckets.items())}


def load_resume_results(resume_path: Optional[str]) -> Dict[str, Dict]:
    if not resume_path:
        return {}
    with open(resume_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    existing = {}
    for r in payload.get('results', []):
        if r.get('judge_correct') is None:
            continue
        key = make_resume_key(
            r.get('image_path', ''),
            r.get('question', ''),
            r.get('ground_truth', ''),
            r.get('question_type', 'unknown'),
        )
        existing[key] = r
    return existing


def evaluate_single_file(
    dataset_name: str,
    dataset_items: list[dict],
    pred_items: list[dict],
    fuzzy_threshold: float,
    judge_url: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_max_workers: int = 8,
    judge_max_tokens: int = 1024,
    judge_max_retries: int = 5,
    existing_judge: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    matched, unmatched_gt, unmatched_pred = match_pred_to_gt(dataset_items, pred_items)

    total = len(dataset_items)
    matched_count = len(matched)
    missing_count = len(unmatched_gt)

    print(f'匹配: {matched_count}/{total}  缺失预测: {missing_count}  多余预测: {len(unmatched_pred)}')

    results = []
    need_judge = []

    overall = init_bucket('overall')
    overall['total'] = total
    overall['matched'] = matched_count
    overall['missing_pred'] = missing_count
    overall['unmatched_pred'] = len(unmatched_pred)

    for idx, (gt_item, pred_item) in enumerate(matched):
        user_msg = next((m for m in gt_item['messages'] if m['role'] == 'user'), {})
        gt_msg = next((m for m in gt_item['messages'] if m['role'] == 'assistant'), {})
        question = extract_question(user_msg.get('content', ''))
        ground_truth = gt_msg.get('content', '')
        prediction = pred_item.get('response', '')
        if prediction is None:
            prediction = ''

        images = gt_item.get('images', [])
        img_path = images[0] if images else ''
        question_type = gt_item.get('question_type', 'unknown')

        match = multi_level_match(prediction, ground_truth, fuzzy_threshold)
        br = compute_bleu_rouge(prediction, ground_truth)

        overall['exact_correct'] += int(match['exact'])
        overall['contain_correct'] += int(match['contain'])
        overall['fuzzy_correct'] += int(match['fuzzy'])
        overall['any_correct'] += int(match['correct_any'])
        overall['sum_bleu'] += br['bleu']
        overall['sum_rouge_l'] += br['rouge_l']

        r = {
            'index': idx,
            'question_type': question_type,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'image_path': img_path,
            'match': match,
            'bleu': br['bleu'],
            'rouge_l': br['rouge_l'],
            'metadata': {
                'iso': gt_item.get('iso'),
                'exp_time': gt_item.get('exp_time'),
                'aperture': gt_item.get('aperture'),
            },
        }

        resume_key = make_resume_key(img_path, question, ground_truth, question_type)
        if match['exact']:
            r['judge_correct'] = True
            r['judge_reason'] = '精确匹配通过，跳过 Judge'
            r['judge_error'] = None
            overall['judge_total'] += 1
            overall['judge_correct'] += 1
        elif existing_judge and resume_key in existing_judge:
            prev = existing_judge[resume_key]
            r['judge_correct'] = prev.get('judge_correct')
            r['judge_reason'] = prev.get('judge_reason')
            r['judge_error'] = prev.get('judge_error')
            if r['judge_correct'] is not None:
                overall['judge_total'] += 1
                overall['judge_correct'] += int(r['judge_correct'] is True)
        elif judge_url and judge_model:
            need_judge.append(r)
        else:
            r['judge_correct'] = None
            r['judge_reason'] = None
            r['judge_error'] = None

        results.append(r)

    for gt_item in unmatched_gt:
        user_msg = next((m for m in gt_item['messages'] if m['role'] == 'user'), {})
        gt_msg = next((m for m in gt_item['messages'] if m['role'] == 'assistant'), {})
        results.append({
            'index': -1,
            'question_type': gt_item.get('question_type', 'unknown'),
            'question': extract_question(user_msg.get('content', '')),
            'ground_truth': gt_msg.get('content', ''),
            'prediction': None,
            'image_path': gt_item.get('images', [''])[0] if gt_item.get('images') else '',
            'match': {'exact': False, 'contain': False, 'fuzzy': False, 'fuzzy_score': 0.0, 'correct_any': False},
            'bleu': 0.0,
            'rouge_l': 0.0,
            'judge_correct': None,
            'judge_reason': '无预测结果',
            'judge_error': None,
        })

    if need_judge:
        print(f'LLM Judge: {len(need_judge)} 条需要判断 '
              f'(精确匹配跳过 {overall["exact_correct"]}, resume 跳过 '
              f'{matched_count - overall["exact_correct"] - len(need_judge)})...')
        judge_lock = threading.Lock()

        def _judge_one(r):
            result = None
            for attempt in range(judge_max_retries + 1):
                result = call_llm_judge(
                    judge_url, judge_model,
                    r['question'], r['ground_truth'], r['prediction'],
                    api_key=judge_api_key, max_tokens=judge_max_tokens,
                )
                if result['judge_error'] is None:
                    break
                if '无法解析' in (result['judge_error'] or ''):
                    break
                if attempt < judge_max_retries:
                    time.sleep(min(2 ** attempt, 30))
            r['judge_correct'] = result['judge_correct']
            r['judge_reason'] = result['judge_reason']
            r['judge_error'] = result['judge_error']
            return r

        with ThreadPoolExecutor(max_workers=judge_max_workers) as executor:
            futures = [executor.submit(_judge_one, r) for r in need_judge]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Judge'):
                r = future.result()
                if r.get('judge_correct') is not None:
                    with judge_lock:
                        overall['judge_total'] += 1
                        overall['judge_correct'] += int(r.get('judge_correct') is True)

    overall_summary = finalize_bucket(overall)
    by_question_type = summarize_by_question_type(results, len(unmatched_pred))

    print(f'精确: {overall_summary["exact_correct"]}/{total} ({overall_summary["exact_accuracy"]:.2%})  '
          f'包含: {overall_summary["contain_correct"]}/{total} ({overall_summary["contain_accuracy"]:.2%})  '
          f'模糊: {overall_summary["fuzzy_correct"]}/{total} ({overall_summary["fuzzy_accuracy"]:.2%})  '
          f'任一: {overall_summary["any_correct"]}/{total} ({overall_summary["any_accuracy"]:.2%})')
    print(f'BLEU: {overall_summary["avg_bleu"]:.4f}  ROUGE-L: {overall_summary["avg_rouge_l"]:.4f}')
    if judge_url and judge_model:
        print(f'Judge: {overall_summary["judge_correct"]}/{overall_summary["judge_total"]} '
              f'({overall_summary["judge_accuracy"]:.2%})')

    return {
        'dataset_name': dataset_name,
        'pred_file': None,
        'dataset_file': None,
        'summary': overall_summary,
        'by_question_type': by_question_type,
        'results': results,
        'unmatched_predictions': unmatched_pred,
    }


def save_report(report: Dict[str, Any], output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = output_dir / f'eval_results_{ts}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'JSON: {json_path}')

    txt_path = output_dir / f'eval_summary_{ts}.txt'
    summary = report['summary']
    by_question_type = report['by_question_type']
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('=' * 110 + '\n')
        f.write('离线评测报告（单文件）\n')
        f.write('=' * 110 + '\n\n')
        f.write(f'时间:         {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'预测文件:     {args.pred_file}\n')
        f.write(f'数据文件:     {args.dataset_file}\n')
        f.write(f'输出目录:     {output_dir}\n')
        f.write(f'Judge URL:    {args.judge_url or "未启用"}\n')
        f.write(f'Resume:       {args.resume or "否"}\n\n')

        f.write('[总体统计]\n')
        f.write(f'总样本:       {summary["total"]}\n')
        f.write(f'匹配预测:     {summary["matched"]}\n')
        f.write(f'缺失预测:     {summary["missing_pred"]}\n')
        f.write(f'多余预测:     {summary["unmatched_pred"]}\n')
        f.write(f'精确匹配:     {summary["exact_correct"]} ({summary["exact_accuracy"]:.2%})\n')
        f.write(f'包含匹配:     {summary["contain_correct"]} ({summary["contain_accuracy"]:.2%})\n')
        f.write(f'模糊匹配:     {summary["fuzzy_correct"]} ({summary["fuzzy_accuracy"]:.2%})\n')
        f.write(f'任一通过:     {summary["any_correct"]} ({summary["any_accuracy"]:.2%})\n')
        f.write(f'平均 BLEU:    {summary["avg_bleu"]:.4f}\n')
        f.write(f'平均 ROUGE-L: {summary["avg_rouge_l"]:.4f}\n')
        if args.judge_url:
            f.write(f'Judge:        {summary["judge_correct"]}/{summary["judge_total"]} '
                    f'({summary["judge_accuracy"]:.2%})\n')
        f.write('\n')

        f.write('[按 question_type 统计]\n')
        header = (
            f'{"question_type":<24} {"total":>6} {"matched":>8} {"missing":>8} '
            f'{"exact":>8} {"contain":>8} {"fuzzy":>8} {"any":>8} {"BLEU":>8} {"ROUGE":>8}'
        )
        if args.judge_url:
            header += f' {"Judge":>12}'
        f.write(header + '\n')
        f.write('-' * 120 + '\n')
        for question_type, bucket in by_question_type.items():
            line = (
                f'{question_type:<24} {bucket["total"]:>6} {bucket["matched"]:>8} {bucket["missing_pred"]:>8} '
                f'{bucket["exact_accuracy"]:>8.2%} {bucket["contain_accuracy"]:>8.2%} '
                f'{bucket["fuzzy_accuracy"]:>8.2%} {bucket["any_accuracy"]:>8.2%} '
                f'{bucket["avg_bleu"]:>8.4f} {bucket["avg_rouge_l"]:>8.4f}'
            )
            if args.judge_url:
                line += f' {bucket["judge_correct"]:>4}/{bucket["judge_total"]:<7}'
            f.write(line + '\n')
        f.write('\n')

        f.write('[错误与缺失概览]\n')
        missing = [r for r in report['results'] if r.get('prediction') is None]
        judge_errors = [r for r in report['results'] if r.get('judge_error')]
        f.write(f'缺失预测条数: {len(missing)}\n')
        f.write(f'Judge 失败条数: {len(judge_errors)}\n')
        if report['unmatched_predictions']:
            f.write(f'多余预测条数: {len(report["unmatched_predictions"])}\n')

    print(f'TXT: {txt_path}')

    csv_path = output_dir / f'eval_results_{ts}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        header = (
            'index,question_type,image_path,question,ground_truth,prediction,'
            'exact,contain,fuzzy,fuzzy_score,correct_any,bleu,rouge_l,judge_correct,judge_error'
        )
        f.write(header + '\n')
        for r in report['results']:
            row = [
                r.get('index'),
                r.get('question_type'),
                r.get('image_path'),
                json.dumps(r.get('question', ''), ensure_ascii=False),
                json.dumps(r.get('ground_truth', ''), ensure_ascii=False),
                json.dumps(r.get('prediction', ''), ensure_ascii=False),
                r.get('match', {}).get('exact', False),
                r.get('match', {}).get('contain', False),
                r.get('match', {}).get('fuzzy', False),
                r.get('match', {}).get('fuzzy_score', 0.0),
                r.get('match', {}).get('correct_any', False),
                r.get('bleu', 0.0),
                r.get('rouge_l', 0.0),
                r.get('judge_correct'),
                json.dumps(r.get('judge_error', ''), ensure_ascii=False),
            ]
            f.write(','.join(map(str, row)) + '\n')
    print(f'CSV: {csv_path}')


def derive_output_dir(pred_file: Path) -> Path:
    return pred_file.parent / 'eval_results_offline'


def main():
    args = parse_args()

    pred_file = Path(args.pred_file)
    dataset_file = Path(args.dataset_file)
    output_dir = Path(args.output_dir) if args.output_dir else derive_output_dir(pred_file)

    print('=' * 60)
    print('离线评测（单文件）')
    print('=' * 60)
    print(f'预测文件: {pred_file}')
    print(f'数据文件: {dataset_file}')
    print(f'输出目录: {output_dir}')
    print(f'Judge URL: {args.judge_url or "未启用"}')
    print(f'Resume:    {args.resume or "否"}')
    print()

    judge_model = None
    if args.judge_url:
        judge_model = args.judge_model or detect_judge_model(args.judge_url, args.judge_api_key)
        if judge_model:
            print(f'Judge 模型: {judge_model}')
        else:
            print('Judge API 无法检测模型，跳过 Judge')
            args.judge_url = None

    existing_judge = load_resume_results(args.resume)
    if existing_judge:
        print(f'加载 Resume judge 结果: {len(existing_judge)} 条')

    dataset_items = load_json_or_jsonl(dataset_file)
    pred_items = load_json_or_jsonl(pred_file)

    print(f'数据条数: {len(dataset_items)}')
    print(f'预测条数: {len(pred_items)}')
    print()

    start_time = time.time()
    report = evaluate_single_file(
        dataset_name=dataset_file.stem,
        dataset_items=dataset_items,
        pred_items=pred_items,
        fuzzy_threshold=args.fuzzy_threshold,
        judge_url=args.judge_url,
        judge_model=judge_model,
        judge_api_key=args.judge_api_key,
        judge_max_workers=args.judge_max_workers,
        judge_max_tokens=args.judge_max_tokens,
        judge_max_retries=args.judge_max_retries,
        existing_judge=existing_judge,
    )
    report['pred_file'] = str(pred_file)
    report['dataset_file'] = str(dataset_file)

    print()
    print('=' * 60)
    print('保存结果')
    print('=' * 60)
    save_report(report, output_dir, args)

    elapsed = time.time() - start_time
    print()
    print('=' * 60)
    print('总体统计')
    print('=' * 60)
    summary = report['summary']
    print(f'总样本:    {summary["total"]}')
    print(f'精确匹配:  {summary["exact_correct"]} ({summary["exact_accuracy"]:.2%})')
    print(f'包含匹配:  {summary["contain_correct"]} ({summary["contain_accuracy"]:.2%})')
    print(f'模糊匹配:  {summary["fuzzy_correct"]} ({summary["fuzzy_accuracy"]:.2%})')
    print(f'任一通过:  {summary["any_correct"]} ({summary["any_accuracy"]:.2%})')
    print(f'avg BLEU:  {summary["avg_bleu"]:.4f}')
    print(f'avg ROU-L: {summary["avg_rouge_l"]:.4f}')
    if args.judge_url:
        print(f'Judge:     {summary["judge_correct"]}/{summary["judge_total"]} ({summary["judge_accuracy"]:.2%})')
    print(f'耗时:      {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print()


if __name__ == '__main__':
    main()
