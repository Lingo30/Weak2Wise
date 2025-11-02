"""
step3_stepmask.py - Step 3: Step-Mask Reasoning Scoring
"""
import os
import json
import math
from tqdm import tqdm
from llm_client import call_llm_text, load_config
from utils import read_jsonl, write_jsonl, normalize_answer_str

def mask_step(step_text: str, fraction_mask: float) -> str:
    if fraction_mask <= 0:
        return step_text
    L = len(step_text)
    m = math.ceil(fraction_mask * L)
    keep = step_text[:max(0, L - m)]
    return keep + " (to be continued...)"

def mask_steps_list(steps, i, n):
    masked = []
    for step in steps:
        if step.strip().lower().startswith("## final answer"):
            masked.append("## Final Answer (to be continued...)")
            continue
        frac = (i / n)
        masked.append(mask_step(step, frac))
    return masked

def steps_list_to_text(steps):
    return "\n\n".join(steps)

def step_mask_score_for_candidate(q, a, step_list, mweak_model, judge_model, n, beta, judge_temperature, majority):
    s_list = []
    for i in range(n):
        masked = mask_steps_list(step_list, i, n)
        hint = steps_list_to_text(masked)
        pqr = load_config()["prompts"]["pqr"].format(question=q, step_reasoning=hint)
        votes = []
        for _ in range(max(1, majority)):
            attempt = call_llm_text(mweak_model, pqr, temperature=0.6)
            votes.append(attempt.strip())
        attempt = max(set(votes), key=votes.count)
        correct = False
        if judge_model:
            judge_prompt = load_config()["prompts"]["pjudge"].format(question=q, attempt=attempt, answer=a)
            judge_out = call_llm_text(judge_model, judge_prompt, temperature=judge_temperature)
            last_line = judge_out.strip().splitlines()[-1].strip() if judge_out.strip().splitlines() else ""
            if last_line.lower() == "yes":
                correct = True
        else:
            if normalize_answer_str(attempt) == normalize_answer_str(a):
                correct = True
        s_list.append(1 if correct else 0)

    savg = sum(s_list) / len(s_list)
    n1 = n - 1
    weighted_sum = 0.0
    for i in range(1, n):
        weighted_sum += s_list[i] * (2 ** (-(n - i)))
    weighted_sum += s_list[0] * (2 ** (-(n - 1)))
    sew = weighted_sum / n1 if n1 > 0 else 0.0
    score = beta * savg + (1 - beta) * sew
    debug = {"s_list": s_list, "savg": savg, "sew": sew}
    return score, debug

def run_step3():
    cfg = load_config()
    work_dir = cfg["dataset"].get("work_dir", "data/work")
    cand_path = os.path.join(work_dir, "candidates.jsonl")
    out_path = os.path.join(work_dir, "scored_candidates.jsonl")
    mweak = cfg["models"]["mweak"]
    judge = cfg["models"].get("judge", None)
    params = cfg["params"]
    n = params.get("n_mask", 6)
    beta = params.get("beta", 0.5)
    judge_temp = params.get("judge_temperature", 0.1)
    majority = params.get("majority", 1)

    results = []
    for rec in tqdm(read_jsonl(cand_path), desc="Step3: scoring"):
        q = rec["question"]
        a = rec["answer"]
        step_list = rec.get("step_list", [])
        if not step_list:
            rec_out = rec.copy()
            rec_out.update({"score": 0.0, "debug": {"s_list": []}})
            results.append(rec_out)
            continue
        score, debug = step_mask_score_for_candidate(q, a, step_list, mweak, judge, n, beta, judge_temp, majority)
        rec_out = rec.copy()
        rec_out.update({"score": score, "debug": debug})
        results.append(rec_out)
    write_jsonl(out_path, results)
    print(f"Step3 done -> {out_path} (total {len(results)})")
    return out_path
