#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ask a running swift deploy service with a local image.')
    parser.add_argument('--image', required=True, help='Local image path to read and send to the service.')
    parser.add_argument('--question', required=True, help='Question text for the image.')
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    parser.add_argument('--model', default='Qwen3-VL-2B-Instruct')
    parser.add_argument('--max-tokens', type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    image_b64 = base64.b64encode(image_path.read_bytes()).decode('utf-8')
    payload = {
        'model': args.model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}},
                {'type': 'text', 'text': args.question},
            ],
        }],
        'max_tokens': args.max_tokens,
        'stream': False,
    }

    response = requests.post(args.url, json=payload, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(f'Service returned HTTP {response.status_code}: {response.text[:2000]}')

    data = response.json()
    try:
        answer = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f'Unexpected response schema: {json.dumps(data, ensure_ascii=False)[:2000]}') from exc
    print(answer.strip())


if __name__ == '__main__':
    main()
