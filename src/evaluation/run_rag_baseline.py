"""
Run the current single-pass RAG answer engine over a fixed benchmark and save metrics.

Use this snapshot before adding agent-style or multi-step retrieval; re-run with the
same questions file to compare quality, latency, and rough cost proxies.

Usage:
  python src/evaluation/run_rag_baseline.py
  python src/evaluation/run_rag_baseline.py --questions data/benchmarks/rag_baseline_questions.jsonl --out data/benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LLM_MODEL, ROOT
from rag.answer_engine import RAGAnswerEngine


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rough_tokens(chars: int) -> float:
    """Very rough token estimate (~4 chars per token for English)."""
    return max(0.0, chars / 4.0)


def _git_revision(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_branch(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_is_dirty(root: Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline benchmark for current RAG")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "rag_baseline_questions.jsonl",
        help="JSONL with id, question, county (optional), difficulty",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "results",
        help="Directory for JSONL + summary JSON",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Retriever top_k (default 8)")
    parser.add_argument(
        "--mode",
        choices=("single", "agent"),
        default="single",
        help="single = one retrieval pass (legacy RAG baseline); agent = decompose + multi-query (default for product comparison use agent)",
    )
    args = parser.parse_args()

    if not args.questions.exists():
        raise FileNotFoundError(f"Missing questions file: {args.questions}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_jsonl = args.out / f"rag_baseline_run_{stamp}.jsonl"
    out_summary = args.out / f"rag_baseline_summary_{stamp}.json"

    items = _load_questions(args.questions)
    engine = RAGAnswerEngine()

    latencies: List[float] = []
    rows_out: List[Dict[str, Any]] = []

    for row in items:
        qid = row["id"]
        question = row["question"]
        county: Optional[str] = row.get("county")
        difficulty = row.get("difficulty", "")

        t0 = time.perf_counter()
        try:
            if args.mode == "agent":
                result = engine.answer_agent(question, county=county, top_k=args.top_k)
            else:
                result = engine.answer_single_pass(question, county=county, top_k=args.top_k)
            err = None
        except Exception as e:
            result = None
            err = str(e)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        rec: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "county": county,
            "difficulty": difficulty,
            "top_k": args.top_k,
            "llm_model": LLM_MODEL,
            "latency_sec": round(elapsed, 4),
            "error": err,
        }

        if result is not None:
            best_d = min((c.distance for c in result.sources), default=None)
            avg_d = (
                statistics.mean(c.distance for c in result.sources)
                if result.sources
                else None
            )
            ctx_chars = sum(len(c.text) for c in result.sources)
            rec.update(
                {
                    "retrieval_mode": getattr(result, "retrieval_mode", args.mode),
                    "confidence": result.confidence,
                    "num_sources": len(result.sources),
                    "best_distance": round(best_d, 6) if best_d is not None else None,
                    "avg_distance": round(avg_d, 6) if avg_d is not None else None,
                    "answer_chars": len(result.answer),
                    "context_chars": ctx_chars,
                    "est_answer_tokens": round(_rough_tokens(len(result.answer)), 1),
                    "est_context_tokens": round(_rough_tokens(ctx_chars), 1),
                    "answer_preview": result.answer[:500]
                    + ("..." if len(result.answer) > 500 else ""),
                }
            )

        rows_out.append(rec)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(args.questions.resolve()),
        "llm_model": LLM_MODEL,
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "n_questions": len(items),
        "n_errors": sum(1 for r in rows_out if r.get("error")),
        "latency_sec_mean": round(statistics.mean(latencies), 4) if latencies else None,
        "latency_sec_median": round(statistics.median(latencies), 4) if latencies else None,
        "latency_sec_stdev": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else None,
        "output_jsonl": str(out_jsonl.resolve()),
        "output_summary": str(out_summary.resolve()),
        "git_commit": _git_revision(ROOT),
        "git_branch": _git_branch(ROOT),
        "git_working_tree_dirty": _git_is_dirty(ROOT),
    }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-question rows: {out_jsonl}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
