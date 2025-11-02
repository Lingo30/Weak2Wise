"""
run_pipeline.py - Orchestrator with CLI and logging

Usage:
    python run_pipeline.py --up-to-step 3
    python run_pipeline.py --up-to-step all

This module exposes run_pipeline(up_to_step: Optional[int]) for programmatic use.
"""
import os
import argparse
import logging
from typing import Optional

from llm_client import load_config
# import step functions
from step1_filter import filter_qa
from step2_generate import generate_candidates
from step3_stepmask import run_step3
from step4_select_truncate import run_step4
from step5_prepare_finetune import prepare_finetune

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging(work_dir: str):
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "pipeline.log")
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # also create a file handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    logging.getLogger().info(f"Logging started. File: {log_file}")

def run_pipeline(up_to_step: Optional[int] = None) -> None:
    """
    Run pipeline steps in order up to `up_to_step`.
    Steps:
      1 -> filter_qa
      2 -> generate_candidates
      3 -> run_step3
      4 -> run_step4
      5 -> prepare_finetune

    If up_to_step is None, run all steps.
    """
    cfg = load_config()
    work_dir = cfg["dataset"].get("work_dir", "data/work")
    setup_logging(work_dir)
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started. up_to_step=%s", up_to_step if up_to_step else "all")

    # translate up_to_step value
    max_step = up_to_step if up_to_step is not None else 5

    try:
        if max_step >= 1:
            logger.info("Running Step 1: filter QA")
            s1_out = filter_qa()
            logger.info("Step 1 finished: %s", s1_out)

        if max_step >= 2:
            logger.info("Running Step 2: generate candidates")
            s2_out = generate_candidates()
            logger.info("Step 2 finished: %s", s2_out)

        if max_step >= 3:
            logger.info("Running Step 3: step-mask scoring")
            s3_out = run_step3()
            logger.info("Step 3 finished: %s", s3_out)

        if max_step >= 4:
            logger.info("Running Step 4: select & truncate")
            s4_out = run_step4()
            logger.info("Step 4 finished: %s", s4_out)

        if max_step >= 5:
            logger.info("Running Step 5: prepare finetune dataset")
            s5_out = prepare_finetune()
            logger.info("Step 5 finished: %s", s5_out)

    except Exception as e:
        logger.exception("Pipeline failed with exception: %s", e)
        raise

    logger.info("Pipeline completed up to step %s", max_step)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run Weak2Wise pipeline (steps 1-5).")
    parser.add_argument(
        "--up-to-step",
        type=str,
        default="all",
        help="Run pipeline up to this step number (1-5) or 'all' to run all steps."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.up_to_step.lower() == "all":
        up_to = None
    else:
        try:
            up_to = int(args.up_to_step)
            if not (1 <= up_to <= 5):
                raise ValueError("up-to-step must be between 1 and 5")
        except Exception as e:
            print("Invalid --up-to-step value. Use 1-5 or 'all'.", e)
            raise
    run_pipeline(up_to_step=up_to)
