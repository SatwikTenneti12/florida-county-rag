"""
Improved rule-based classifier (keyword + topic-specific context).

This aims to improve overall accuracy by:
- Detecting topic-specific action terms (surveys/inventories/assessments; crossings infrastructure)
- Requiring wildlife context for ambiguous topics (surveys/crossings)
- Adding negative filters for common non-wildlife uses of words like "crossing"

No LLM calls. Deterministic and reproducible.

Output:
  data/processed/topic_evidence_by_county.csv (same location/shape as metrics.py expects)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "topic_evidence_by_county.csv"

# Topic keys must match src/evaluation/metrics.py
TOPIC_KEYS = ["wildlife_corridors", "wildlife_crossings", "land_acquisition", "wildlife_surveys", "open_space"]

WILDLIFE_CONTEXT_RE = re.compile(
    r"\b(wildlife|fauna|animal|animals|species|habitat)\b",
    re.IGNORECASE,
)

# Crossings: wildlife passage infrastructure
WILDLIFE_CROSSING_INFRA_RE = re.compile(
    r"\b(wildlife crossing|wildlife underpass|wildlife overpass|ecopassage|fauna passage|wildlife bridge|culvert|underpass|overpass|crossing structure|wildlife crossing)\b",
    re.IGNORECASE,
)

# Common false-positive phrases for "crossings"
CROSSING_NEGATIVE_RE = re.compile(
    r"\b(pedestrian crossing|crosswalk|crossing guard|road crossing|railroad crossing|train crossing)\b",
    re.IGNORECASE,
)

# Surveys: survey/inventory/assessment/monitoring language
WILDLIFE_SURVEY_ACTION_RE = re.compile(
    r"\b(wildlife survey|wildlife surveys|species survey|habitat survey|biological survey|ecological survey|baseline survey|preconstruction survey|wildlife inventory|species inventory|biological inventory|ecological assessment|biological assessment|ecological assessment|wildlife monitoring|species monitoring)\b",
    re.IGNORECASE,
)

# Corridors: keep classic phrases
WILDLIFE_CORRIDOR_RE = re.compile(
    r"(wildlife corridor|corridor linkage|ecological corridor|habitat corridor|movement corridor)",
    re.IGNORECASE,
)

# Land acquisition + open space: keep simple & high precision
LAND_ACQUISITION_RE = re.compile(
    r"(land acquisition|acquire land|conservation acquisition|purchase land|fee simple acquisition|conservation easement)",
    re.IGNORECASE,
)

OPEN_SPACE_RE = re.compile(
    r"(open space|greenway|green space|conservation area|natural area|preservation area|open lands)",
    re.IGNORECASE,
)


def classify_chunk(text: str) -> dict[str, bool]:
    t = text or ""

    # Normalize a little for consistent regex performance
    # (do not remove punctuation; keep it for multi-word phrases).
    # Regex is case-insensitive anyway.
    wildlife_ctx = bool(WILDLIFE_CONTEXT_RE.search(t))

    # Corridors
    corridors = bool(WILDLIFE_CORRIDOR_RE.search(t))

    # Crossings
    crossings = False
    if not CROSSING_NEGATIVE_RE.search(t) and wildlife_ctx:
        crossings = bool(WILDLIFE_CROSSING_INFRA_RE.search(t))

    # Surveys
    surveys = False
    if wildlife_ctx:
        surveys = bool(WILDLIFE_SURVEY_ACTION_RE.search(t))

    # Land acquisition + open space
    land_acq = bool(LAND_ACQUISITION_RE.search(t))
    open_space = bool(OPEN_SPACE_RE.search(t))

    return {
        "wildlife_corridors": corridors,
        "wildlife_crossings": crossings,
        "land_acquisition": land_acq,
        "wildlife_surveys": surveys,
        "open_space": open_space,
    }


def detect_topics() -> None:
    best = defaultdict(lambda: {k: False for k in TOPIC_KEYS})

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            county = r["county"]
            text = r["text"]

            for topic, hit in classify_chunk(text).items():
                if hit:
                    best[county][topic] = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["county"] + TOPIC_KEYS)
        for county in sorted(best.keys()):
            row = [county] + [best[county][t] for t in TOPIC_KEYS]
            writer.writerow(row)

    print("Done.")
    print("Output:", OUT_PATH)
    print("Counties processed:", len(best))


if __name__ == "__main__":
    detect_topics()

