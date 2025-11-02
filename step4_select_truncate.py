"""
step4_select_truncate.py - Step 4: Golden Reasoning Selection and Truncation
"""
import os
from llm_client import call_llm_text, load_config
from utils import read_jsonl, write_jsonl, count_tokens_simple, normalize_answer_str

def truncate_normal_heuristic(normal_text: str, correct_answer: str, trunc_model: str = None, trunc_temp: float = 0.1):
    if not normal_text:
        return normal_text
    prompt = load_config()["prompts"]["ptrunc"].format(
        normal_reasoning=normal_text,
        answer=correct_answer
    )
    out = call_llm_text(trunc_model, prompt, temperature=trunc_temp)
    return out.strip()

def run_step4():
    cfg = load_config()
    work_dir = cfg["dataset"].get("work_dir", "data/work")
    scored_path = os.path.join(work_dir, "scored_candidates.jsonl")
    out_path = os.path.join(work_dir, "selected_truncated.jsonl")
    trunc_model = cfg["models"].get("truncation", None)
    trunc_temp = cfg["params"].get("trunc_temp", 0.1)

    if not trunc_model:
        raise ValueError("LLM-based truncation requires 'truncation' model in config.")

    agg = {}
    for rec in read_jsonl(scored_path):
        key = (rec["question"], rec["answer"])
        agg.setdefault(key, []).append(rec)

    output = []
    for (q, a), cand_list in agg.items():
        best_score = max([c.get("score", 0.0) for c in cand_list])
        best_cands = [c for c in cand_list if c.get("score", 0.0) == best_score]
        if len(best_cands) == 1:
            chosen = best_cands[0]
        else:
            chosen = min(best_cands, key=lambda c: count_tokens_simple(c.get("normal", "")))
        truncated = truncate_normal_heuristic(chosen.get("normal", ""), a, trunc_model=trunc_model, trunc_temp=trunc_temp)
        out = {
            "question": q,
            "answer": a,
            "chosen_raw": chosen.get("raw", ""),
            "chosen_normal": chosen.get("normal", ""),
            "truncated_normal": truncated,
            "score": chosen.get("score", 0.0),
            "debug": chosen.get("debug", {})
        }
        output.append(out)
    write_jsonl(out_path, output)
    print(f"Step4 done -> {out_path} (selected {len(output)})")
    return out_path
