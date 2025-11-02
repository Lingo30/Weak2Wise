"""
utils.py - Common utility functions
"""
import re
import json
import os
from typing import List, Dict, Any, Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_answer_str(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip('. ')
    return s.lower()

def extract_step_reasoning_from_assistant(text: str):
    if not text:
        return []
    parts = re.split(r'(?m)^(?=## )', text)
    steps = []
    for p in parts:
        p = p.strip()
        if p.startswith("##"):
            steps.append(p)
    return steps

def count_tokens_simple(text: str) -> int:
    if not text:
        return 0
    return len(text.split())
