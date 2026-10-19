"""
Evaluation: recall@K, MRR, wrong-variant@1 on 40 verified HVAC questions.

Two metrics reported:
  doc_hit: doc_id in top-K (primary; catches wrong-variant failures)
  ibm_hit: expected_answer substring in any top-K chunk text (Pipeline's metric, for comparability)

Usage:
    python src/eval.py
    python src/eval.py --top-k 5 --strategy best_fields --out results/eval_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

EVAL_PATH = Path("data/eval/eval_set_verified40.csv")

def _load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))

def _doc_hit(results: list[dict], gold_doc: str) -> tuple[bool, int]:
    for rank, r in enumerate(results, start=1):
        if r["doc_id"] == gold_doc:
            return True, rank
    return False, 0

def _ibm_substring_hit(results: list[dict], expected_answer: str) -> bool:
    if not expected_answer:
        return False
    ea = expected_answer.strip().lower()
    return any(ea in r.get("text", "").lower() for r in results)

def _wrong_variant_at_1(results: list[dict], gold_doc: str) -> bool:
    return bool(results) and results[0]["doc_id"] != gold_doc

def run_eval(top_k: int = 5, strategy: str = "best_fields", use_sparse: bool = True, out_path: Path | None = None) -> dict:
    questions = _load_questions(EVAL_PATH)
    logger.info(f"Loaded {len(questions)} questions  top_k={top_k}  strategy={strategy}")

    by_type: dict[str, list] = defaultdict(list)
    records = []

    for i, q in enumerate(questions):
        query = q["question"]
        qtype = q["question_type"]
        gold_doc = q["doc_id"]
        exp_answer = q.get("expected_answer", "")

        logger.info(f"[{i+1}/{len(questions)}] {qtype}: {query[:70]}")

        try:
            results = retrieve(query, top_k=top_k, strategy=strategy, use_sparse=use_sparse)
        except Exception as e:
            logger.error(f"  Error: {e}")
            results = []

        doc_hit, doc_rank = _doc_hit(results, gold_doc)
        ibm_hit = _ibm_substring_hit(results, exp_answer)
        wrong_v1 = _wrong_variant_at_1(results, gold_doc)
        rr = 1.0 / doc_rank if doc_rank > 0 else 0.0

        record = {
            "question": query,
            "type": qtype,
            "gold_doc": gold_doc,
            "doc_hit": doc_hit,
            "ibm_hit": ibm_hit,
            "doc_rank": doc_rank,
            "rr": rr,
            "wrong_v1": wrong_v1,
            "top1_doc": results[0]["doc_id"] if results else None,
        }
        records.append(record)
        by_type[qtype].append(record)

    def _agg(recs: list[dict]) -> dict:
        n = len(recs)
        return {
            "n": n,
            "doc_recall@k": sum(r["doc_hit"] for r in recs) / n,
            "ibm_recall@k": sum(r["ibm_hit"] for r in recs) / n,
            "mrr": sum(r["rr"] for r in recs) / n,
            "wrong_v1_rate": sum(r["wrong_v1"] for r in recs) / n,
        }

    overall     = _agg(records)
    by_type_agg = {t: _agg(rs) for t, rs in by_type.items()}

    summary = {
        "top_k":    top_k,
        "strategy": strategy,
        "overall":  overall,
        "by_type":  by_type_agg,
        "records":  records,
    }

    print(f"\n{'='*60}")
    print(f"  top_k={top_k}  strategy={strategy}")
    print(f"  {'─'*56}")
    print(f"  {'Metric':<20} {'Overall':>8}  {'lookup':>8}  {'reasoning':>10}  {'trap':>6}")
    print(f"  {'─'*56}")
    for metric in ["doc_recall@k", "ibm_recall@k", "mrr", "wrong_v1_rate"]:
        row = f"  {metric:<20} {overall[metric]:>8.3f}"
        for t in ["lookup", "reasoning", "trap"]:
            m = by_type_agg.get(t, {})
            row += f"  {m.get(metric, 0):>8.3f}"
        print(row)
    print(f"{'='*60}")
    print("  Note: ibm_recall = substring match (IBM metric); doc_recall = document-level hit (ours)\n")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Results → {out_path}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", choices=["match", "best_fields"], default="best_fields")
    parser.add_argument("--no-sparse", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/eval_results.json"))
    args = parser.parse_args()
    run_eval(args.top_k, args.strategy, use_sparse=not args.no_sparse, out_path=args.out)
