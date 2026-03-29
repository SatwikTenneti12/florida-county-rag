"""
Deterministic post-processing for RAG predictions.

Goal:
Reduce false positives by enforcing that LLM key_quotes for each topic
actually contain topic-action terms.

This does NOT call the LLM; it only post-processes existing JSONL output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any


ROOT = Path(__file__).resolve().parents[2]
IN_JSONL = ROOT / "data" / "processed" / "rag_topic_evidence_by_county.jsonl"
OUT_CSV = ROOT / "data" / "processed" / "rag_topic_evidence_by_county_postprocessed.csv"

TOPICS = [
    "wildlife_corridors",
    "wildlife_crossings",
    "land_acquisition",
    "wildlife_surveys",
    "open_space",
]

ACTION_PATTERNS = {
    "wildlife_corridors": re.compile(
        r"\b(wildlife corridor|ecological corridor|habitat corridor|movement corridor|corridor linkage)\b",
        re.IGNORECASE,
    ),
    "wildlife_crossings": re.compile(
        r"\b(wildlife crossing|wildlife underpass|wildlife overpass|ecopassage|fauna passage|wildlife bridge|culvert|underpass|overpass)\b",
        re.IGNORECASE,
    ),
    "land_acquisition": re.compile(
        r"\b(land acquisition|acquire land|purchase land|conservation acquisition|conservation easement|fee simple acquisition)\b",
        re.IGNORECASE,
    ),
    "wildlife_surveys": re.compile(
        r"\b(wildlife survey|wildlife surveys|species survey|habitat survey|biological survey|baseline survey|preconstruction survey|wildlife inventory|species inventory|wildlife monitoring|species monitoring)\b",
        re.IGNORECASE,
    ),
    "open_space": re.compile(
        r"\b(open space|greenway|green space|conservation area|natural area|preservation area|open lands)\b",
        re.IGNORECASE,
    ),
}


def join_text(record: Dict[str, Any]) -> str:
    """Collect text evidence to check for action terms."""
    parts: list[str] = []
    parts.append(str(record.get("evidence_summary", "")))
    for q in record.get("key_quotes", []) or []:
        parts.append(str(q))
    # sources aren't cheap to parse; key_quotes are enough for guardrails
    return "\n".join(parts).lower()


def postprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    topic = record.get("topic")
    if topic not in TOPICS:
        return record

    # If the LLM says it has policy but the key_quotes don't mention the
    # actual required action, force it to False.
    if record.get("has_policy") is True:
        evidence_text = join_text(record)
        action_re = ACTION_PATTERNS.get(topic)
        if action_re and not action_re.search(evidence_text):
            record["has_policy"] = False
            record["confidence"] = "low"
            record["policy_type"] = "none"
            record["evidence_summary"] = (
                "Post-processed: key quotes did not include topic-action terms required for this classification."
            )
            record["key_quotes"] = record.get("key_quotes", []) or []

    return record


def main() -> None:
    if not IN_JSONL.exists():
        raise FileNotFoundError(f"Missing input JSONL: {IN_JSONL}")

    # Build results: results[county][topic] = bool
    results: Dict[str, Dict[str, bool]] = {}

    with IN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = postprocess_record(rec)

            county = rec["county"]
            topic = rec["topic"]
            results.setdefault(county, {})[topic] = bool(rec.get("has_policy", False))

    # Write CSV in the same shape as metrics.py expects
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        header = ["county"] + TOPICS
        f_out.write(",".join(header) + "\n")

        for county in sorted(results.keys()):
            row = [county] + [str(results[county].get(t, False)) for t in TOPICS]
            f_out.write(",".join(row) + "\n")

    print(f"Wrote post-processed CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()

