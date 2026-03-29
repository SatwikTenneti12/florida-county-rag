"""
Supervised threshold/classifier approach for topic evidence.

This uses the existing labeled dataset (county_labels.csv) to learn
topic-specific decision thresholds from deterministic features derived from
vector retrieval + topic-action keyword matches.

Features per (county, topic):
- min_distance among top-K retrieved chunks
- mean_distance among top-K retrieved chunks
- num_action_chunks among top-K retrieved chunks (topic-action regex hits)
- num_keyword_chunks among top-K retrieved chunks (topic keyword hits)

Model:
- LogisticRegression per topic (5-fold stratified CV to pick a threshold)

Output:
- data/processed/topic_evidence_by_county_ml.csv
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TOPICS, LABELS_PATH
from utils.county_normalizer import normalize_county_name
from utils.embeddings import embed_query

import chromadb
from chromadb.config import Settings


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
PRED_OUT_PATH = ROOT / "data" / "processed" / "topic_evidence_by_county_ml.csv"

COLLECTION_NAME = "county_chunks"
CHROMA_DIR = ROOT / "chroma_db"

TOPIC_ACTION_RE: Dict[str, re.Pattern] = {
    "wildlife_corridors": re.compile(r"(wildlife corridor|corridor linkage|ecological corridor|habitat corridor|movement corridor)", re.IGNORECASE),
    "wildlife_crossings": re.compile(
        r"(wildlife crossing|wildlife underpass|wildlife overpass|ecopassage|fauna passage|wildlife bridge|culvert|underpass|overpass|crossing structure)",
        re.IGNORECASE,
    ),
    "land_acquisition": re.compile(
        r"(land acquisition|acquire land|purchase land|conservation acquisition|fee simple acquisition|conservation easement)",
        re.IGNORECASE,
    ),
    "wildlife_surveys": re.compile(
        r"(wildlife survey|wildlife surveys|species survey|habitat survey|biological survey|ecological survey|baseline survey|preconstruction survey|wildlife inventory|species inventory|biological inventory|ecological assessment|biological assessment|wildlife monitoring|species monitoring)",
        re.IGNORECASE,
    ),
    "open_space": re.compile(
        r"(open space|greenway|green space|conservation area|natural area|preservation area|open lands)",
        re.IGNORECASE,
    ),
}

# Simple wildlife context constraint for crossings/surveys to reduce generic false matches.
WILDLIFE_CTX_RE = re.compile(r"\b(wildlife|fauna|animal|animals|species|habitat)\b", re.IGNORECASE)
NEGATIVE_CROSSING_RE = re.compile(r"\b(pedestrian crossing|crosswalk|crossing guard|road crossing|railroad crossing|train crossing)\b", re.IGNORECASE)
ENFORCEABLE_RE = re.compile(r"\b(shall|must|required|require[s]?)\b", re.IGNORECASE)
SURVEY_CONTEXT_RE = re.compile(
    r"\b(before|prior to|prior-to|as a condition|development|construction|project|permit|site plan|zoning|land use)\b",
    re.IGNORECASE,
)

# Wildlife survey evidence in comprehensive plans often appears as:
# "As a condition of permit approval ... survey of the development site ... presence of species"
WILDLIFE_SURVEY_EVIDENCE_WORD_RE = re.compile(
    r"\b(survey|inventory|monitoring|assessment|evaluation)\b",
    re.IGNORECASE,
)
WILDLIFE_SURVEY_CONDITION_RE = re.compile(
    r"\b(as a condition of (permit )?approval|permit approval|development site|development review|prior to|before)\b",
    re.IGNORECASE,
)


def bool_to_yesno(x: bool) -> str:
    return "True" if x else "False"


def to_bool(x: str) -> bool:
    return (x or "").strip().lower() == "yes"


def load_labels() -> Dict[str, Dict[str, bool]]:
    """Return labels[county][topic_key] = bool."""
    # metrics.py uses topic mapping to read columns; we mirror that here.
    topic_cols = {
        "wildlife_corridors": "Wildlife Corridors",
        "land_acquisition": "Land Acquisition",
        "wildlife_crossings": "Wildlife Crossings",
        "wildlife_surveys": "Wildlife Surveys",
        "open_space": "Open Space Planning",
    }

    labels: Dict[str, Dict[str, bool]] = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            county = row["County"].strip()
            labels[county] = {k: to_bool(row.get(col, "")) for k, col in topic_cols.items()}
    return labels


def get_chunk_county_aliases() -> Dict[str, str]:
    """
    Map canonical label county name -> actual Chroma/Chunk county string.
    Uses normalization to match.
    """
    # Collect unique county strings found in chunks
    chunk_counties = set()
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_counties.add(rec["county"])

    chunk_counties_list = sorted(chunk_counties)

    mapping: Dict[str, str] = {}
    for label_county in load_labels().keys():
        nl = normalize_county_name(label_county)
        for c in chunk_counties_list:
            if normalize_county_name(c).lower() == nl.lower():
                mapping[label_county] = c
                break

    return mapping


@dataclass
class FeatureRow:
    county: str
    topic: str
    y: int
    min_dist: float
    mean_dist: float
    min_dist_action: float
    min_dist_action_enforceable: float
    num_action_chunks: int
    num_action_enforceable_chunks: int
    num_keyword_chunks: int
    num_keyword_enforceable_chunks: int


def extract_chunk_signal(
    topic_key: str, chunk_text: str
) -> Tuple[bool, bool, bool, bool]:
    """
    Returns:
      (action_hit, keyword_hit, action_enforceable_hit, keyword_enforceable_hit)
    """
    t = chunk_text or ""
    tl = t.lower()
    # wildlife_surveys needs a richer matcher than TOPIC_ACTION_RE because many plans
    # use "survey of the development site" rather than "wildlife survey".
    if topic_key == "wildlife_surveys":
        action_hit = (
            bool(WILDLIFE_CTX_RE.search(tl))
            and bool(WILDLIFE_SURVEY_EVIDENCE_WORD_RE.search(tl))
            and bool(WILDLIFE_SURVEY_CONDITION_RE.search(tl))
        )
    else:
        action_re = TOPIC_ACTION_RE.get(topic_key)
        action_hit = bool(action_re.search(tl)) if action_re else False

    enforceable_hit = bool(ENFORCEABLE_RE.search(tl))

    if topic_key in ("wildlife_crossings", "wildlife_surveys"):
        # Require some wildlife context and avoid common non-wildlife crossings.
        if not WILDLIFE_CTX_RE.search(tl):
            action_hit = False
        if topic_key == "wildlife_crossings" and NEGATIVE_CROSSING_RE.search(tl):
            action_hit = False

    keywords = TOPICS[topic_key]["keywords"][:]
    keyword_hit = any((k.lower() in tl) for k in keywords)

    if topic_key in ("wildlife_crossings", "wildlife_surveys"):
        if not WILDLIFE_CTX_RE.search(tl):
            keyword_hit = False
        if topic_key == "wildlife_crossings" and NEGATIVE_CROSSING_RE.search(tl):
            keyword_hit = False

        # For wildlife_surveys, treat the broader evidence matcher as the "keyword" signal too,
        # so the model can learn from these plans even when exact keyword phrases vary.
        if topic_key == "wildlife_surveys" and action_hit:
            keyword_hit = True

    action_enforceable_hit = action_hit and enforceable_hit
    keyword_enforceable_hit = keyword_hit and enforceable_hit

    return action_hit, keyword_hit, action_enforceable_hit, keyword_enforceable_hit


def build_features_for_topic(
    topic_key: str,
    label_map: Dict[str, Dict[str, bool]],
    county_aliases: Dict[str, str],
    col,
    query_embeddings: List[List[float]],
    top_k: int,
) -> List[FeatureRow]:
    rows: List[FeatureRow] = []

    for county, topic_labels in label_map.items():
        y = 1 if topic_labels[topic_key] else 0
        chunk_county = county_aliases.get(county)

        # If we cannot map the county name to the chunk metadata, skip.
        if not chunk_county:
            continue

        # Multi-query retrieval for higher recall. We merge and deduplicate
        # by chunk id, keeping the lowest distance per chunk.
        per_query_k = max(8, min(15, top_k))
        combined: Dict[str, Tuple[str, float]] = {}

        for qe in query_embeddings:
            res = col.query(
                query_embeddings=[qe],
                n_results=per_query_k,
                where={"county": chunk_county},
                include=["documents", "distances"],
            )
            docs = res["documents"][0] or []
            dists = res["distances"][0] or []
            ids = res.get("ids", [[]])[0] or []

            for doc, dist, cid in zip(docs, dists, ids):
                if cid not in combined or float(dist) < combined[cid][1]:
                    combined[cid] = (doc or "", float(dist))

        merged = sorted(combined.values(), key=lambda x: x[1])[:top_k]
        docs = [d for d, _ in merged]
        dists = [dist for _, dist in merged]

        if not dists:
            min_dist = 999.0
            mean_dist = 999.0
        else:
            min_dist = float(min(dists))
            mean_dist = float(sum(dists) / len(dists))

        min_dist_action = 999.0
        min_dist_action_enforceable = 999.0

        num_action_chunks = 0
        num_action_enforceable_chunks = 0
        num_keyword_chunks = 0
        num_keyword_enforceable_chunks = 0

        for i, doc in enumerate(docs):
            dist_i = float(dists[i]) if i < len(dists) else 999.0
            action_hit, keyword_hit, action_enforceable_hit, keyword_enforceable_hit = extract_chunk_signal(
                topic_key, doc
            )
            if action_hit:
                num_action_chunks += 1
                min_dist_action = min(min_dist_action, dist_i)
            if keyword_hit:
                num_keyword_chunks += 1
            if action_enforceable_hit:
                num_action_enforceable_chunks += 1
                min_dist_action_enforceable = min(min_dist_action_enforceable, dist_i)
            if keyword_enforceable_hit:
                num_keyword_enforceable_chunks += 1

        rows.append(
            FeatureRow(
                county=county,
                topic=topic_key,
                y=y,
                min_dist=min_dist,
                mean_dist=mean_dist,
                min_dist_action=min_dist_action,
                min_dist_action_enforceable=min_dist_action_enforceable,
                num_action_chunks=num_action_chunks,
                num_action_enforceable_chunks=num_action_enforceable_chunks,
                num_keyword_chunks=num_keyword_chunks,
                num_keyword_enforceable_chunks=num_keyword_enforceable_chunks,
            )
        )

    return rows


def train_and_predict_topic(
    topic_key: str,
    rows: List[FeatureRow],
    seed: int = 42,
) -> Dict[str, float]:
    """
    Train logistic regression with CV to pick threshold,
    then fit full model and return probabilities.
    """
    # Build X, y
    X = np.array(
        [
            [
                r.min_dist,
                r.mean_dist,
                r.min_dist_action,
                r.min_dist_action_enforceable,
                r.num_action_chunks,
                r.num_action_enforceable_chunks,
                r.num_keyword_chunks,
                r.num_keyword_enforceable_chunks,
            ]
            for r in rows
        ],
        dtype=float,
    )
    y = np.array([r.y for r in rows], dtype=int)
    counties = [r.county for r in rows]

    # If one class only, predict that class for everyone.
    if len(set(y.tolist())) < 2:
        p = float(y.mean())
        return {c: p for c in counties}

    # OOF probabilities for threshold selection
    oof_probs = np.zeros(len(rows), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        clf.fit(X_train, y_train)
        oof_probs[test_idx] = clf.predict_proba(X_test)[:, 1]

    # Choose threshold that maximizes accuracy on OOF
    thresholds = np.linspace(0.05, 0.95, 19)  # coarse grid
    best_thr = 0.5
    best_acc = -1.0
    for thr in thresholds:
        y_hat = (oof_probs >= thr).astype(int)
        acc = float((y_hat == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    # Fit final model and return probabilities; downstream can apply threshold
    clf_final = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    clf_final.fit(X, y)
    probs = clf_final.predict_proba(X)[:, 1]

    return {c: float(p) for c, p in zip(counties, probs)}, best_thr


def main():
    labels = load_labels()
    county_aliases = get_chunk_county_aliases()

    # Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION_NAME)

    top_k = 25
    seed = 42

    # Precompute embeddings for multi-query retrieval per topic.
    # For each topic we use: [topic query] + first 3 keyword phrases.
    topic_embeddings: Dict[str, List[List[float]]] = {}
    for topic_key in TOPICS.keys():
        query_variants = [TOPICS[topic_key]["query"]] + TOPICS[topic_key]["keywords"][:3]
        topic_embeddings[topic_key] = [embed_query(q) for q in query_variants]

    # Build dataset and train per topic
    # Store final boolean predictions
    final_preds: Dict[str, Dict[str, bool]] = {}

    # We'll pick thresholds per topic from OOF and apply them on full fit.
    topic_thresholds: Dict[str, float] = {}

    for topic_key in TOPICS.keys():
        rows = build_features_for_topic(
            topic_key=topic_key,
            label_map=labels,
            county_aliases=county_aliases,
            col=col,
            query_embeddings=topic_embeddings[topic_key],
            top_k=top_k,
        )

        # Train and get probabilities + threshold
        result = train_and_predict_topic(topic_key, rows, seed=seed)
        probs_by_county, best_thr = result
        topic_thresholds[topic_key] = best_thr

        for county in labels.keys():
            final_preds.setdefault(county, {})
            p = probs_by_county.get(county, 0.0)
            final_preds[county][topic_key] = p >= best_thr

    # Export CSV compatible with metrics.py
    PRED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRED_OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["county"] + list(TOPICS.keys())
        writer.writerow(header)
        for county in sorted(labels.keys()):
            row = [county] + [final_preds[county].get(t, False) for t in TOPICS.keys()]
            writer.writerow(row)

    print("Done.")
    print("Output:", PRED_OUT_PATH)
    print("Thresholds:", {k: round(v, 3) for k, v in topic_thresholds.items()})


if __name__ == "__main__":
    main()

