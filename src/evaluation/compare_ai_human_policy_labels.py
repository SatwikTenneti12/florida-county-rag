"""
Compare AI-generated policy theme scores against human-coded county labels.

The human workbook contains yes/no labels for five conservation themes. The AI
rubric scorer writes 0-3 scores for the same themes. For agreement metrics, AI
scores greater than 0 are treated as "present" and score 0 as "absent".
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.county_normalizer import find_matching_county

PROCESSED_DIR = ROOT / "data" / "processed"

TOPIC_COLUMNS = {
    "wildlife_corridors": "Wildlife Corridors",
    "land_acquisition": "Land Acquisition",
    "wildlife_crossings": "Wildlife Crossings",
    "wildlife_surveys": "Wildlife Surveys",
    "open_space": "Open Space Planning",
}


def clean_yes_no(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return 1
    if text in {"no", "n", "false", "0"}:
        return 0
    return None


def parse_rank_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"\b([123])\s*/\s*3\b", text)
    if match:
        return int(match.group(1))
    return None


def load_human_labels(xlsx_path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read the human coding workbook") from exc

    df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        county_raw = row.get("County")
        if not isinstance(county_raw, str) or not county_raw.strip():
            continue
        county = find_matching_county(county_raw.strip()) or county_raw.strip()
        out: Dict[str, Any] = {
            "county": county,
            "student": row.get("Student"),
            "year_adopted": row.get("Year Adopted"),
            "year_ending": row.get("Year Ending"),
            "year_amendment": row.get("Year Amendment"),
            "human_rank_score": parse_rank_score(row.get("Ranking")),
            "ranking_raw": row.get("Ranking"),
            "ranking_rationale": row.get("Ranking Rationale"),
        }
        for topic, column in TOPIC_COLUMNS.items():
            out[f"{topic}_human"] = clean_yes_no(row.get(column))
        rows.append(out)

    return rows


def write_human_outputs(rows: List[Dict[str, Any]], csv_path: Path, summary_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "county",
        "student",
        "year_adopted",
        "year_ending",
        "year_amendment",
        *[f"{topic}_human" for topic in TOPIC_COLUMNS],
        "human_rank_score",
        "ranking_raw",
        "ranking_rationale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    topic_summary = {}
    for topic in TOPIC_COLUMNS:
        values = [r[f"{topic}_human"] for r in rows if r[f"{topic}_human"] is not None]
        yes = sum(1 for v in values if v == 1)
        no = sum(1 for v in values if v == 0)
        topic_summary[topic] = {
            "yes": yes,
            "no": no,
            "total_labeled": len(values),
            "yes_percent": round((yes / len(values)) * 100, 2) if values else 0,
        }

    rank_values = [r["human_rank_score"] for r in rows if r["human_rank_score"] is not None]
    summary = {
        "human_rows": len(rows),
        "topic_summary": topic_summary,
        "all_five_yes_count": sum(
            1
            for r in rows
            if all(r[f"{topic}_human"] == 1 for topic in TOPIC_COLUMNS)
        ),
        "all_five_yes_counties": [
            r["county"]
            for r in rows
            if all(r[f"{topic}_human"] == 1 for topic in TOPIC_COLUMNS)
        ],
        "ranked_rows_with_explicit_score": len(rank_values),
        "explicit_rank_distribution": {
            str(score): sum(1 for v in rank_values if v == score)
            for score in [1, 2, 3]
        },
        "note": (
            "The workbook contains yes/no theme labels for 67 county rows. "
            "The Ranking column has explicit 1/3, 2/3, or 3/3 scores only in "
            "some rows; other rows contain date-like values and should not be "
            "used as rank scores without cleanup."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_ai_scores(ai_csv_path: Path) -> Dict[str, Dict[str, int]]:
    if not ai_csv_path.exists():
        return {}
    rows = {}
    with ai_csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            county = find_matching_county(row.get("county", "")) or row.get("county", "")
            rows[county] = {}
            for topic in TOPIC_COLUMNS:
                value = row.get(f"{topic}_score")
                try:
                    rows[county][topic] = int(float(value))
                except (TypeError, ValueError):
                    rows[county][topic] = 0
    return rows


def compare(rows: List[Dict[str, Any]], ai_scores: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    comparisons = []
    for human in rows:
        county = human["county"]
        ai = ai_scores.get(county)
        if not ai:
            continue
        for topic in TOPIC_COLUMNS:
            human_value = human[f"{topic}_human"]
            if human_value is None:
                continue
            ai_score = ai.get(topic, 0)
            comparisons.append(
                {
                    "county": county,
                    "topic": topic,
                    "human_present": human_value,
                    "ai_score": ai_score,
                    "ai_present": 1 if ai_score > 0 else 0,
                }
            )

    def cohen_kappa(items: List[Dict[str, Any]]) -> Optional[float]:
        if not items:
            return None
        n = len(items)
        observed = sum(1 for c in items if c["human_present"] == c["ai_present"]) / n
        human_yes = sum(1 for c in items if c["human_present"] == 1) / n
        human_no = 1 - human_yes
        ai_yes = sum(1 for c in items if c["ai_present"] == 1) / n
        ai_no = 1 - ai_yes
        expected = (human_yes * ai_yes) + (human_no * ai_no)
        if expected == 1:
            return 1.0 if observed == 1 else None
        return (observed - expected) / (1 - expected)

    topic_metrics = {}
    for topic in TOPIC_COLUMNS:
        items = [c for c in comparisons if c["topic"] == topic]
        if not items:
            topic_metrics[topic] = {"n": 0}
            continue
        tp = sum(1 for c in items if c["human_present"] == 1 and c["ai_present"] == 1)
        tn = sum(1 for c in items if c["human_present"] == 0 and c["ai_present"] == 0)
        fp = sum(1 for c in items if c["human_present"] == 0 and c["ai_present"] == 1)
        fn = sum(1 for c in items if c["human_present"] == 1 and c["ai_present"] == 0)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        topic_metrics[topic] = {
            "n": len(items),
            "agreement": round((tp + tn) / len(items), 4),
            "cohen_kappa": round(cohen_kappa(items), 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    return {
        "matched_counties": len({c["county"] for c in comparisons}),
        "comparison_rows": len(comparisons),
        "overall_agreement": round(
            sum(1 for c in comparisons if c["human_present"] == c["ai_present"])
            / len(comparisons),
            4,
        )
        if comparisons
        else None,
        "overall_cohen_kappa": round(cohen_kappa(comparisons), 4) if comparisons else None,
        "topic_metrics": topic_metrics,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AI policy scores with human labels")
    parser.add_argument(
        "--human-xlsx",
        type=Path,
        default=ROOT / "data" / "County Comprehensive Plans.xlsx",
    )
    parser.add_argument(
        "--ai-csv",
        type=Path,
        default=PROCESSED_DIR / "ai_policy_scores_by_county.csv",
    )
    parser.add_argument(
        "--human-out",
        type=Path,
        default=PROCESSED_DIR / "human_policy_labels.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROCESSED_DIR / "human_policy_label_summary.json",
    )
    parser.add_argument(
        "--agreement-out",
        type=Path,
        default=PROCESSED_DIR / "ai_human_policy_agreement.json",
    )
    args = parser.parse_args()

    human_rows = load_human_labels(args.human_xlsx)
    write_human_outputs(human_rows, args.human_out, args.summary_out)

    ai_scores = load_ai_scores(args.ai_csv)
    agreement = compare(human_rows, ai_scores)
    args.agreement_out.write_text(json.dumps(agreement, indent=2), encoding="utf-8")

    print(f"Wrote human labels: {args.human_out}")
    print(f"Wrote human summary: {args.summary_out}")
    print(f"Wrote AI-human agreement: {args.agreement_out}")
    print(f"Human county rows: {len(human_rows)}")
    print(f"Matched AI counties: {agreement['matched_counties']}")


if __name__ == "__main__":
    main()
