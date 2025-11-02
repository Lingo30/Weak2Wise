"""
step2_generate.py - Step 2: Candidate Reasoning Trace Generation
"""
import os
import json
from tqdm import tqdm
from llm_client import call_llm_chat, load_config
from utils import read_jsonl, write_jsonl, extract_step_reasoning_from_assistant

def generate_candidates():
    cfg = load_config()
    data_cfg = cfg["dataset"]
    work_dir = data_cfg.get("work_dir", "data/work")
    os.makedirs(work_dir, exist_ok=True)
    s_prime_path = os.path.join(work_dir, "s_prime.jsonl")
    out_path = os.path.join(work_dir, "candidates.jsonl")

    models = cfg["models"]["mstrong"]
    samples = cfg["params"].get("samples_per_model", 3)
    temp = cfg["params"].get("gen_temperature", 0.6)
    cgen = cfg["prompts"]["cgen"]

    results = []
    for rec in tqdm(read_jsonl(s_prime_path), desc="Step2: gen"):
        q = rec["question"]
        a = rec["answer"]
        for m in models:
            for _ in range(samples):
                messages = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                    {"role": "user", "content": cgen}
                ]
                resp = call_llm_chat(m, messages, temperature=temp)
                raw = resp.get("raw", None)
                step_reasoning = resp.get("content", "")
                reasoning_content = resp.get("reasoning_content")
                normal = reasoning_content
                steps = extract_step_reasoning_from_assistant(step_reasoning)
                results.append({
                    "question": q,
                    "answer": a,
                    "m_strong": m,
                    "raw": raw,
                    "step_reasoning": step_reasoning,
                    "step_list": steps,
                    "normal": normal
                })
    write_jsonl(out_path, results)
    print(f"Step2 done. candidates -> {out_path} (total {len(results)})")
    return out_path
