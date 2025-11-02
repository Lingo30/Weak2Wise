"""
step5_prepare_finetune.py - Step 5: Fine-Tuning Data Preparation
"""
import os
from llm_client import load_config
from utils import read_jsonl, write_jsonl

def prepare_finetune():
    cfg = load_config()
    work_dir = cfg["dataset"].get("work_dir", "data/work")
    in_path = os.path.join(work_dir, "selected_truncated.jsonl")
    out_path = cfg["dataset"].get("finetune_out", os.path.join(work_dir, "finetune.jsonl"))
    records = []
    for rec in read_jsonl(in_path):
        q = rec["question"]
        a = rec["answer"]
        r = rec.get("truncated_normal", "")
        response = f"<think> {r.strip()} </think> {a.strip()}"
        records.append({"prompt": q.strip(), "response": response})
    write_jsonl(out_path, records)
    print(f"Step5 done -> {out_path} (size {len(records)})")
    return out_path
