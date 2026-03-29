#!/usr/bin/env python3
"""
Command-line interface for the RAG answer engine.

Usage:
    python -m src.rag.cli "What wildlife corridor policies exist in Alachua County?"
    python -m src.rag.cli "Does any county require wildlife surveys?" --top-k 15
    python -m src.rag.cli "Open space requirements" --county "Palm Beach County"
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.answer_engine import RAGAnswerEngine


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about Florida county comprehensive plans"
    )
    parser.add_argument(
        "question",
        type=str,
        help="Your question about county policies"
    )
    parser.add_argument(
        "--county",
        type=str,
        default=None,
        help="Filter to a specific county"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve (default: 8)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text"
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Only show retrieved sources without generating an answer"
    )
    
    args = parser.parse_args()
    
    try:
        engine = RAGAnswerEngine()
        
        if args.sources_only:
            # Just retrieve and show sources
            from rag.retriever import get_retriever
            retriever = get_retriever()
            chunks = retriever.retrieve(
                args.question,
                top_k=args.top_k,
                county_filter=args.county
            )
            
            print(f"\nQuery: {args.question}")
            if args.county:
                print(f"County filter: {args.county}")
            print(f"\nFound {len(chunks)} relevant excerpts:\n")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"[{i}] {chunk.citation} (distance: {chunk.distance:.4f})")
                snippet = chunk.text[:400].replace("\n", " ")
                print(f"    {snippet}...")
                print()
            return
        
        # Full RAG answer
        result = engine.answer(
            args.question,
            county=args.county,
            top_k=args.top_k
        )
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "=" * 80)
            print(f"QUESTION: {result.question}")
            if result.county_filter:
                print(f"COUNTY: {result.county_filter}")
            print(f"CONFIDENCE: {result.confidence}")
            print("=" * 80)
            print()
            print(result.format_with_citations())
            print()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
