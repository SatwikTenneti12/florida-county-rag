"""
Comprehensive evaluation metrics for topic classification.

Computes precision, recall, F1, and provides detailed error analysis.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LABELS_PATH, PROCESSED_DIR, TOPICS
from utils.county_normalizer import normalize_county_name, find_matching_county


@dataclass
class ClassificationMetrics:
    """Metrics for a single classification task."""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def total(self) -> int:
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4)
        }


@dataclass
class ErrorCase:
    """Details of a classification error."""
    county: str
    topic: str
    label: bool
    prediction: bool
    error_type: str  # "false_positive" or "false_negative"
    confidence: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "county": self.county,
            "topic": self.topic,
            "label": self.label,
            "prediction": self.prediction,
            "error_type": self.error_type,
            "confidence": self.confidence
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    topic_metrics: Dict[str, ClassificationMetrics] = field(default_factory=dict)
    overall_metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    errors: List[ErrorCase] = field(default_factory=list)
    matched_counties: int = 0
    total_label_counties: int = 0
    unmatched_counties: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            "",
            f"Counties matched: {self.matched_counties}/{self.total_label_counties}",
            f"Unmatched: {len(self.unmatched_counties)}",
            "",
            "OVERALL METRICS:",
            f"  Accuracy:  {self.overall_metrics.accuracy:.2%}",
            f"  Precision: {self.overall_metrics.precision:.2%}",
            f"  Recall:    {self.overall_metrics.recall:.2%}",
            f"  F1 Score:  {self.overall_metrics.f1:.2%}",
            "",
            "METRICS BY TOPIC:",
            "-" * 60,
        ]
        
        for topic, metrics in self.topic_metrics.items():
            lines.extend([
                f"  {topic}:",
                f"    Accuracy:  {metrics.accuracy:.2%}  ({metrics.true_positives + metrics.true_negatives}/{metrics.total})",
                f"    Precision: {metrics.precision:.2%}  Recall: {metrics.recall:.2%}  F1: {metrics.f1:.2%}",
                f"    TP: {metrics.true_positives}  TN: {metrics.true_negatives}  FP: {metrics.false_positives}  FN: {metrics.false_negatives}",
                ""
            ])
        
        lines.extend([
            "-" * 60,
            f"ERRORS ({len(self.errors)} total):",
        ])
        
        # Group errors by type
        fp_errors = [e for e in self.errors if e.error_type == "false_positive"]
        fn_errors = [e for e in self.errors if e.error_type == "false_negative"]
        
        lines.append(f"\n  False Positives ({len(fp_errors)}):")
        for e in fp_errors[:10]:
            lines.append(f"    - {e.county}: {e.topic}")
        if len(fp_errors) > 10:
            lines.append(f"    ... and {len(fp_errors) - 10} more")
        
        lines.append(f"\n  False Negatives ({len(fn_errors)}):")
        for e in fn_errors[:10]:
            lines.append(f"    - {e.county}: {e.topic}")
        if len(fn_errors) > 10:
            lines.append(f"    ... and {len(fn_errors) - 10} more")
        
        if self.unmatched_counties:
            lines.extend([
                "",
                f"UNMATCHED COUNTIES ({len(self.unmatched_counties)}):",
                "  " + ", ".join(self.unmatched_counties[:15]),
            ])
            if len(self.unmatched_counties) > 15:
                lines.append(f"  ... and {len(self.unmatched_counties) - 15} more")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall_metrics.to_dict(),
            "by_topic": {t: m.to_dict() for t, m in self.topic_metrics.items()},
            "errors": [e.to_dict() for e in self.errors],
            "matched_counties": self.matched_counties,
            "total_label_counties": self.total_label_counties,
            "unmatched_counties": self.unmatched_counties
        }


class Evaluator:
    """Evaluate predictions against ground truth labels."""
    
    TOPIC_COLUMN_MAP = {
        "wildlife_corridors": "Wildlife Corridors",
        "land_acquisition": "Land Acquisition", 
        "wildlife_crossings": "Wildlife Crossings",
        "wildlife_surveys": "Wildlife Surveys",
        "open_space": "Open Space Planning",
    }
    
    def __init__(self, labels_path: Path = LABELS_PATH):
        self.labels = self._load_labels(labels_path)
    
    def _load_labels(self, path: Path) -> Dict[str, Dict[str, bool]]:
        """Load ground truth labels with normalized county names."""
        labels = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                county = row["County"].strip()
                labels[county] = {}
                
                for topic_key, col_name in self.TOPIC_COLUMN_MAP.items():
                    value = row.get(col_name, "").strip().lower()
                    labels[county][topic_key] = value == "yes"
        
        return labels
    
    def evaluate(
        self,
        predictions_path: Path,
        use_normalized_matching: bool = True
    ) -> EvaluationReport:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions_path: Path to predictions CSV
            use_normalized_matching: Try to match county names by normalization
            
        Returns:
            EvaluationReport with detailed metrics
        """
        # Load predictions
        preds = {}
        pred_confidence = {}
        
        with predictions_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                county = row["county"].strip()
                preds[county] = {}
                pred_confidence[county] = {}
                
                for topic in TOPICS.keys():
                    value = str(row.get(topic, "False")).strip().lower()
                    preds[county][topic] = value == "true"
                    
                    conf_key = f"{topic}_confidence"
                    if conf_key in row:
                        pred_confidence[county][topic] = row[conf_key]
        
        # Match predictions to labels
        report = EvaluationReport()
        report.total_label_counties = len(self.labels)
        
        topic_metrics = {t: ClassificationMetrics() for t in TOPICS.keys()}
        overall = ClassificationMetrics()
        errors = []
        matched_counties = []
        
        for label_county, label_topics in self.labels.items():
            # Find matching prediction
            pred_county = self._find_matching_pred(label_county, preds, use_normalized_matching)
            
            if pred_county is None:
                report.unmatched_counties.append(label_county)
                continue
            
            matched_counties.append(label_county)
            pred_topics = preds[pred_county]
            
            for topic in TOPICS.keys():
                if topic not in label_topics:
                    continue
                
                label = label_topics[topic]
                pred = pred_topics.get(topic, False)
                confidence = pred_confidence.get(pred_county, {}).get(topic)
                
                # Update metrics
                if label and pred:  # True Positive
                    topic_metrics[topic].true_positives += 1
                    overall.true_positives += 1
                elif not label and not pred:  # True Negative
                    topic_metrics[topic].true_negatives += 1
                    overall.true_negatives += 1
                elif pred and not label:  # False Positive
                    topic_metrics[topic].false_positives += 1
                    overall.false_positives += 1
                    errors.append(ErrorCase(
                        county=label_county,
                        topic=topic,
                        label=label,
                        prediction=pred,
                        error_type="false_positive",
                        confidence=confidence
                    ))
                else:  # False Negative
                    topic_metrics[topic].false_negatives += 1
                    overall.false_negatives += 1
                    errors.append(ErrorCase(
                        county=label_county,
                        topic=topic,
                        label=label,
                        prediction=pred,
                        error_type="false_negative",
                        confidence=confidence
                    ))
        
        report.matched_counties = len(matched_counties)
        report.topic_metrics = topic_metrics
        report.overall_metrics = overall
        report.errors = errors
        
        return report
    
    def _find_matching_pred(
        self,
        label_county: str,
        preds: Dict[str, Any],
        use_normalized: bool
    ) -> Optional[str]:
        """Find the prediction county that matches the label county."""
        # Exact match
        if label_county in preds:
            return label_county
        
        if not use_normalized:
            return None
        
        # Try normalized matching
        normalized_label = normalize_county_name(label_county)
        
        for pred_county in preds:
            normalized_pred = normalize_county_name(pred_county)
            if normalized_label.lower() == normalized_pred.lower():
                return pred_county
        
        # Try partial matching (county name without "County")
        label_base = normalized_label.lower().replace(" county", "").strip()
        for pred_county in preds:
            pred_base = normalize_county_name(pred_county).lower().replace(" county", "").strip()
            if label_base == pred_base:
                return pred_county
        
        return None


def main():
    """Run evaluation on prediction files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate topic classification predictions")
    parser.add_argument(
        "predictions",
        type=str,
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(LABELS_PATH),
        help="Path to ground truth labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable county name normalization"
    )
    
    args = parser.parse_args()
    
    evaluator = Evaluator(Path(args.labels))
    report = evaluator.evaluate(
        Path(args.predictions),
        use_normalized_matching=not args.no_normalize
    )
    
    print(report.summary())
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nWrote JSON report to {args.output}")


if __name__ == "__main__":
    main()
