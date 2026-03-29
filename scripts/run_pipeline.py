#!/usr/bin/env python3
"""
Run the full RAG pipeline:
1. Smart chunking (re-chunk with sentence awareness)
2. Rebuild vector index
3. Run RAG classifier on all counties
4. Evaluate against baseline

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --chunk --index
    python scripts/run_pipeline.py --classify --evaluate
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=ROOT)
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully")
        return True
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Florida County RAG pipeline")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--chunk", action="store_true", help="Run smart chunking")
    parser.add_argument("--index", action="store_true", help="Rebuild vector index")
    parser.add_argument("--classify", action="store_true", help="Run RAG classifier")
    parser.add_argument("--classify-baseline", action="store_true", help="Run keyword classifier")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate results")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    
    args = parser.parse_args()
    
    # If no specific step, show help
    if not any([args.all, args.chunk, args.index, args.classify, args.classify_baseline, args.evaluate]):
        parser.print_help()
        return
    
    steps = []
    
    if args.all or args.chunk:
        steps.append((
            [sys.executable, "src/ingestion/smart_chunker.py"],
            "Smart Chunking (sentence-aware)"
        ))
    
    if args.all or args.index:
        steps.append((
            [sys.executable, "src/indexing/build_chroma.py", "--batch", str(args.batch_size)],
            "Building Vector Index"
        ))
    
    if args.all or args.classify_baseline:
        steps.append((
            [sys.executable, "src/classification/rule_classifier.py"],
            "Keyword-based Classification (baseline)"
        ))
    
    if args.all or args.classify:
        steps.append((
            [sys.executable, "src/classification/rag_classifier.py", "--detailed"],
            "RAG-based Classification"
        ))
    
    if args.all or args.evaluate:
        # Evaluate keyword baseline
        steps.append((
            [sys.executable, "src/evaluation/metrics.py", "data/processed/topic_evidence_by_county.csv"],
            "Evaluate Keyword Baseline"
        ))
        # Evaluate RAG classifier (if it exists)
        rag_results = ROOT / "data" / "processed" / "rag_topic_evidence_by_county.csv"
        if rag_results.exists() or args.classify or args.all:
            steps.append((
                [sys.executable, "src/evaluation/metrics.py", str(rag_results)],
                "Evaluate RAG Classifier"
            ))
    
    # Run all steps
    results = []
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {desc}")
    
    failed = sum(1 for _, s in results if not s)
    if failed:
        print(f"\n{failed} step(s) failed")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} steps completed successfully!")


if __name__ == "__main__":
    main()
