"""
Compare rule-based detections against spreadsheet labels.

Inputs:
- data/manifests/county_labels.csv
- data/processed/topic_evidence_by_county.csv

Outputs:
- data/processed/eval_joined.csv
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data" / "manifests" / "county_labels.csv"
PRED_PATH = ROOT / "data" / "processed" / "topic_evidence_by_county.csv"
OUT_PATH = ROOT / "data" / "processed" / "eval_joined.csv"


TOPIC_MAP = {
    "wildlife_corridors": "Wildlife Corridors",
    "land_acquisition": "Land Acquisition",
    "wildlife_crossings": "Wildlife Crossings",
    "wildlife_surveys": "Wildlife Surveys",
    "open_space": "Open Space Planning",
}


def to_bool(x: str) -> bool:
    return (x or "").strip().lower() == "yes"


def main():
    # load labels
    labels = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            county = row["County"].strip()
            labels[county] = row

    # load predictions
    preds = {}
    with PRED_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            county = row["county"].strip()
            preds[county] = row

    # evaluate
    results = {t: {"correct": 0, "total": 0} for t in TOPIC_MAP}
    mismatches = []

    joined_rows = []
    for county, lab_row in labels.items():
        if county not in preds:
            continue

        pred_row = preds[county]

        out_row = {"County": county}
        for topic, col in TOPIC_MAP.items():
            y = to_bool(lab_row.get(col, ""))
            p = str(pred_row.get(topic, "False")).strip().lower() == "true"

            out_row[f"label_{topic}"] = y
            out_row[f"pred_{topic}"] = p
            out_row[f"match_{topic}"] = (y == p)

            results[topic]["total"] += 1
            if y == p:
                results[topic]["correct"] += 1
            else:
                mismatches.append((county, topic, y, p))

        joined_rows.append(out_row)

    # write joined CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(joined_rows[0].keys()) if joined_rows else ["County"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(joined_rows)

    print("Accuracy by topic:")
    for topic in TOPIC_MAP:
        total = results[topic]["total"]
        correct = results[topic]["correct"]
        acc = correct / total if total else 0
        print(f"- {topic}: {correct}/{total} ({acc:.2%})")

    print("\nTotal mismatches:", len(mismatches))
    print("First 10 mismatches:")
    for m in mismatches[:10]:
        print(m)

    print("\nWrote eval table:", OUT_PATH)


if __name__ == "__main__":
    main()
