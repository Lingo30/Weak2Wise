"""
step1_filter.py - Step 1: Question-Answer Data Filtering
"""
import os
from typing import List, Tuple
from tqdm import tqdm
import json
from llm_client import call_llm_chat, call_llm_text, load_config
from utils import read_jsonl, write_jsonl, normalize_answer_str

def filter_qa():
    cfg = load_config()
    data_cfg = cfg["dataset"]
    work_dir = data_cfg.get("work_dir", "data/work")
    os.makedirs(work_dir, exist_ok=True)
    in_path = data_cfg["input_path"]
    mweak = cfg["models"]["mweak"]
    judge_model = cfg["models"].get("judge", None)
    params = cfg["params"]
    use_judge = judge_model is not None and not params.get("skip_filter", False)

    out_path = os.path.join(work_dir, "s_prime.jsonl")
    retained = []
    for rec in tqdm(read_jsonl(in_path), desc="Step1: filtering"):
        q = rec.get("question")
        a = rec.get("answer")
        if not q or a is None:
            continue
        # query weak model
        try:
            resp = call_llm_chat(mweak, [{"role": "user", "content": q}], temperature=params.get("gen_temperature", 0.6))
            attempt = resp.get("content", "").strip()
        except Exception as e:
            raise

        correct_flag = False
        if use_judge:
            judge_prompt = cfg["prompts"]["pjudge"].format(question=q, attempt=attempt, answer=a)
            judge_out = call_llm_text(judge_model, judge_prompt, temperature=params.get("judge_temperature", 0.1))
            last_line = judge_out.strip().splitlines()[-1].strip() if judge_out.strip().splitlines() else ""
            if last_line.lower() == "yes":
                correct_flag = True
        else:
            if normalize_answer_str(attempt) == normalize_answer_str(a):
                correct_flag = True

        if not correct_flag:
            retained.append({"question": q, "answer": a, "weak_attempt": attempt})

    write_jsonl(out_path, retained)
    print(f"Step1 done. retained {len(retained)} -> {out_path}")
    return out_path
