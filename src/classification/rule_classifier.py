"""
Rule-based topic detection baseline.

Output:
data/processed/topic_evidence_by_county.csv
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "topic_evidence_by_county.csv"


TOPICS = {
    # Keywords here are intentionally broader (recall),
    # but we only mark a topic as present when we also see enforceable
    # policy language near the keyword (precision).
    "wildlife_corridors": [
        "wildlife corridor",
        "corridor linkage",
        "ecological corridor",
        "habitat corridor",
        "movement corridor",
        "wildlife movement corridor",
    ],
    "wildlife_crossings": [
        "wildlife crossing",
        "wildlife underpass",
        "wildlife overpass",
        "wildlife crossing structure",
        "wildlife bridge",
        "ecopassage",
        "fauna crossing",
        "animal crossing",
        # Note: "crossing" alone is too ambiguous, so we keep it mostly
        # as part of specific phrases above.
    ],
    "land_acquisition": [
        "land acquisition",
        "acquire land",
        "acquisition of land",
        "conservation acquisition",
        "purchase land",
        "purchase of land",
        "fee simple acquisition",
        "conservation easement",
    ],
    "wildlife_surveys": [
        "wildlife survey",
        "wildlife surveys",
        "species survey",
        "habitat survey",
        "biological survey",
        "ecological survey",
        "wildlife inventory",
        "species inventory",
        "biological inventory",
        "ecological assessment",
        "biological assessment",
        "species monitoring",
        "wildlife monitoring",
        "preconstruction survey",
        "baseline survey",
    ],
    "open_space": [
        "open space",
        "greenway",
        "green space",
        "conservation area",
        "natural area",
        "preservation area",
        "open lands",
    ],
}

ENFORCEABLE_RE = re.compile(r"\b(shall|must|required|required|require[s]?)\b", re.IGNORECASE)

# Phrases that often create false positives if found near "wildlife crossing".
NEGATIVE_BY_TOPIC = {
    "wildlife_crossings": [
        "pedestrian crossing",
        "crosswalk",
        "crossing guard",
        "road crossing",
        "railroad crossing",
        "train crossing",
    ]
}

# For wildlife crossings, also require the context to mention wildlife/fauna/animals.
CROSSING_ANCHOR_RE = re.compile(r"\b(wildlife|fauna|animal|animals|species)\b", re.IGNORECASE)


def mark_if_enforceable_near_keyword(
    text_lower: str,
    *,
    keywords: list[str],
    negative_phrases: list[str] | None = None,
    require_anchor_re: re.Pattern | None = None,
    window_before: int = 250,
    window_after: int = 500,
) -> bool:
    """
    Only return True when:
      1) a topic keyword occurs, AND
      2) enforceable policy language ("shall/must/required/...") occurs
         within a nearby character window, AND
      3) optional negatives/anchors are satisfied.
    """
    negative_phrases = negative_phrases or []

    for kw in keywords:
        if kw not in text_lower:
            continue

        # Evaluate around the first occurrence; if not enforceable, we still
        # keep scanning other occurrences.
        start = 0
        while True:
            idx = text_lower.find(kw, start)
            if idx == -1:
                break

            ctx_start = max(0, idx - window_before)
            ctx_end = min(len(text_lower), idx + window_after)
            ctx = text_lower[ctx_start:ctx_end]

            if negative_phrases and any(neg in ctx for neg in negative_phrases):
                start = idx + len(kw)
                continue

            if require_anchor_re and not require_anchor_re.search(ctx):
                start = idx + len(kw)
                continue

            if ENFORCEABLE_RE.search(ctx):
                return True

            start = idx + len(kw)

    return False


def detect_topics():
    best = defaultdict(lambda: {t: False for t in TOPICS})

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            county = r["county"]
            text = r["text"].lower()

            for topic, keywords in TOPICS.items():
                if best[county][topic]:
                    continue

                if mark_if_enforceable_near_keyword(
                    text,
                    keywords=keywords,
                    negative_phrases=NEGATIVE_BY_TOPIC.get(topic),
                    require_anchor_re=CROSSING_ANCHOR_RE if topic == "wildlife_crossings" else None,
                ):
                    best[county][topic] = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["county"] + list(TOPICS.keys()))

        for county in sorted(best.keys()):
            row = [county] + [best[county][t] for t in TOPICS]
            writer.writerow(row)

    print("Done.")
    print("Output:", OUT_PATH)
    print("Counties processed:", len(best))


if __name__ == "__main__":
    detect_topics()