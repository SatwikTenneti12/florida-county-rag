"""
RAG-based 0-3 rubric scoring for county comprehensive plan policy strength.

This script implements the manuscript scoring task:
- five conservation policy themes
- 0-3 score per county/topic
- composite score out of 15
- percentage and Weak/Moderate/Strong tier

Output:
    data/processed/ai_policy_scores_by_county.csv
    data/processed/ai_policy_scores_by_county.jsonl
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHUNKS_PATH, PROCESSED_DIR, TOPICS
from rag.retriever import RetrievedChunk, VectorRetriever, get_retriever
from utils.llm import get_llm_client
from classification.rag_classifier import ACTION_PATTERNS, ENFORCEABLE_RE


RUBRIC_PROMPT = """You are scoring a Florida county comprehensive plan for conservation policy strength.

COUNTY: {county}
TOPIC: {topic_name}
TOPIC KEY: {topic_key}
TOPIC DESCRIPTION: {topic_description}

Score the retrieved excerpts using this 0-3 rubric:

0 = Absent
No reference to the category concept, by any terminology.

1 = Minimal
The concept is acknowledged or mentioned in passing, but no conservation goal, standard, or implementation mechanism is attached.

2 = Substantial
The concept is explicitly addressed with a stated conservation purpose and at least one identifiable standard, mapped area, procedural requirement, or implementation mechanism, but the language is largely advisory, discretionary, or inconsistently applied.

3 = Robust
The concept is explicitly and recurrently addressed with enforceable standards, clear decision triggers, assigned responsibilities, and/or spatial specificity.

DOCUMENT EXCERPTS:
{context}

Important scoring rules:
- Score only from the excerpts provided. Do not assume policy content not shown here.
- Score 0 only when the provided excerpts contain no clear reference to the topic concept or an equivalent term.
- If the excerpts clearly mention the concept but do not attach a conservation goal, standard, or mechanism, score 1 rather than 0.
- Treat "shall", "must", "required", and "require" as evidence of enforceability.
- Treat "should", "encourage", "consider", "may", and broad goals/objectives as weaker advisory language unless paired with clear requirements.
- Wildlife crossings require wildlife passage infrastructure or explicit passage/crossing mitigation, not ordinary pedestrian/road/railroad crossings.
- Wildlife surveys require survey, inventory, biological assessment, monitoring, or equivalent site evaluation language for wildlife/species/habitat, not generic habitat protection alone.
- Open space should be scored for conservation/open-space planning relevance; purely recreational open space with no conservation or land-use mechanism should not receive a high score.

Return ONLY valid JSON with this exact shape:
{{
  "score": 0,
  "confidence": "high",
  "policy_strength": "absent",
  "rationale": "One concise paragraph explaining the score.",
  "supporting_excerpt_ids": ["E1"],
  "key_quotes": ["Short quote from retrieved text"]
}}

Allowed values:
- score: 0, 1, 2, or 3
- confidence: "high", "medium", or "low"
- policy_strength: "absent", "minimal", "substantial", or "robust"
"""


@dataclass
class RubricScore:
    county: str
    topic: str
    score: int
    confidence: str
    policy_strength: str
    rationale: str
    key_quotes: List[str]
    supporting_excerpt_ids: List[str]
    source_chunks: List[RetrievedChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "county": self.county,
            "topic": self.topic,
            "score": self.score,
            "confidence": self.confidence,
            "policy_strength": self.policy_strength,
            "rationale": self.rationale,
            "key_quotes": self.key_quotes,
            "supporting_excerpt_ids": self.supporting_excerpt_ids,
            "sources": [c.to_dict() for c in self.source_chunks],
        }


def get_unique_counties_from_chunks() -> List[str]:
    counties = set()
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            county = row.get("county")
            if county and not county.startswith("_"):
                counties.add(county)
    return sorted(counties)


def score_to_strength(score: int) -> str:
    return {
        0: "absent",
        1: "minimal",
        2: "substantial",
        3: "robust",
    }.get(score, "absent")


def composite_tier(composite_score: int) -> str:
    if composite_score <= 5:
        return "Weak"
    if composite_score <= 10:
        return "Moderate"
    return "Strong"


class RAGRubricScorer:
    def __init__(self, retriever: Optional[VectorRetriever] = None):
        self.retriever = retriever or get_retriever()
        self.llm = get_llm_client()
        self._chunk_cache: Dict[str, List[Dict[str, Any]]] = {}

    def score_topic(self, county: str, topic_key: str, top_k: int = 8) -> RubricScore:
        topic_config = TOPICS[topic_key]
        chunks = self._retrieve_evidence(county, topic_key, top_k=top_k)

        if not chunks:
            return RubricScore(
                county=county,
                topic=topic_key,
                score=0,
                confidence="high",
                policy_strength="absent",
                rationale="No relevant excerpts were retrieved for this topic.",
                key_quotes=[],
                supporting_excerpt_ids=[],
                source_chunks=[],
            )

        prompt = RUBRIC_PROMPT.format(
            county=county,
            topic_name=topic_config["display_name"],
            topic_key=topic_key,
            topic_description=topic_config["description"],
            context=self._build_context(chunks),
        )

        response = self.llm.complete(prompt, temperature=0.0, max_tokens=900)
        parsed = self._parse_score_response(response)
        score = int(parsed.get("score", 0))
        score = max(0, min(3, score))

        return RubricScore(
            county=county,
            topic=topic_key,
            score=score,
            confidence=str(parsed.get("confidence", "low")).lower(),
            policy_strength=str(parsed.get("policy_strength", score_to_strength(score))).lower(),
            rationale=str(parsed.get("rationale", "")).strip(),
            key_quotes=[str(q).strip() for q in parsed.get("key_quotes", []) if str(q).strip()],
            supporting_excerpt_ids=[
                str(e).strip()
                for e in parsed.get("supporting_excerpt_ids", [])
                if str(e).strip()
            ],
            source_chunks=chunks,
        )

    def score_county(
        self,
        county: str,
        topics: Optional[List[str]] = None,
        top_k: int = 8,
    ) -> Dict[str, RubricScore]:
        if topics is None:
            topics = list(TOPICS.keys())
        return {topic: self.score_topic(county, topic, top_k=top_k) for topic in topics}

    def _retrieve_evidence(self, county: str, topic_key: str, top_k: int) -> List[RetrievedChunk]:
        topic_config = TOPICS[topic_key]
        queries = [topic_config["query"]] + topic_config["keywords"]
        top_k_per_query = max(8, min(16, top_k * 2))

        semantic_chunks = self.retriever.retrieve_multi_query(
            queries=queries,
            top_k_per_query=top_k_per_query,
            county_filter=county,
            deduplicate=True,
        )
        lexical_chunks = self._lexical_evidence(county, topic_key, limit=max(top_k, 10))
        chunks = self._merge_chunks(semantic_chunks + lexical_chunks)

        action_re = ACTION_PATTERNS.get(topic_key)

        def rank_chunk(chunk: RetrievedChunk) -> float:
            text = chunk.text or ""
            score = 0.0
            if action_re and action_re.search(text):
                score += 8.0
            if ENFORCEABLE_RE.search(text):
                score += 3.0
            if topic_key == "wildlife_crossings" and re.search(
                r"\b(pedestrian|crosswalk|crossing guard|railroad crossing|train crossing)\b",
                text,
                re.IGNORECASE,
            ):
                score -= 5.0
            score += max(0.0, 1.0 - float(chunk.distance))
            return score

        ranked = sorted(chunks, key=rank_chunk, reverse=True)
        return ranked[:top_k]

    def _lexical_evidence(self, county: str, topic_key: str, limit: int) -> List[RetrievedChunk]:
        topic_config = TOPICS[topic_key]
        phrases = self._topic_phrases(topic_config)
        if not phrases:
            return []

        scored_chunks = []
        for row in self._county_rows(county):
            text = row.get("text") or ""
            text_l = text.lower()
            phrase_score = 0
            exact_hits = 0

            for phrase in phrases:
                hits = text_l.count(phrase)
                if hits:
                    exact_hits += hits
                    phrase_score += hits * (8 if " " in phrase else 3)

            if exact_hits == 0:
                continue

            action_re = ACTION_PATTERNS.get(topic_key)
            if action_re and action_re.search(text):
                phrase_score += 8
            if ENFORCEABLE_RE.search(text):
                phrase_score += 4
            if re.search(r"\b(conservation|preserv|protect|natural|habitat|greenway|easement)\b", text, re.IGNORECASE):
                phrase_score += 3

            scored_chunks.append((phrase_score, row))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        chunks = []
        for score, row in scored_chunks[:limit]:
            text = self._focused_text(row.get("text") or "", phrases)
            chunk_id = (
                f"lexical:{row.get('county')}:{row.get('pdf_file')}:"
                f"{row.get('page_start')}:{row.get('page_end')}:{row.get('chunk_index')}"
            )
            chunks.append(
                RetrievedChunk(
                    text=text,
                    county=row.get("county") or county,
                    pdf_file=row.get("pdf_file") or "Unknown",
                    page_start=int(row.get("page_start") or 0),
                    page_end=int(row.get("page_end") or 0),
                    distance=1.0 / (1.0 + score),
                    chunk_id=chunk_id,
                )
            )
        return chunks

    @staticmethod
    def _focused_text(text: str, phrases: List[str], window: int = 900) -> str:
        text_l = text.lower()
        match_positions = [text_l.find(phrase) for phrase in phrases if text_l.find(phrase) >= 0]
        if not match_positions:
            return text

        center = min(match_positions)
        start = max(0, center - window // 2)
        end = min(len(text), center + window)

        while start > 0 and text[start] not in ".\n;":
            start -= 1
        while end < len(text) and text[end - 1] not in ".\n;":
            end += 1

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet

    @staticmethod
    def _topic_phrases(topic_config: Dict[str, Any]) -> List[str]:
        terms = [topic_config.get("query", "")] + list(topic_config.get("keywords", []))
        phrases = set()
        for term in terms:
            for cleaned in re.findall(r"[a-zA-Z][a-zA-Z/-]*(?:\s+[a-zA-Z][a-zA-Z/-]*)*", term.lower()):
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if len(cleaned) >= 4:
                    phrases.add(cleaned)
        return sorted(phrases, key=len, reverse=True)

    def _county_rows(self, county: str) -> List[Dict[str, Any]]:
        if county not in self._chunk_cache:
            rows = []
            with CHUNKS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("county") == county:
                        rows.append(row)
            self._chunk_cache[county] = rows
        return self._chunk_cache[county]

    @staticmethod
    def _merge_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        merged: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = "|".join(
                [
                    chunk.county,
                    chunk.pdf_file,
                    str(chunk.page_start),
                    str(chunk.page_end),
                    chunk.text[:160],
                ]
            )
            current = merged.get(key)
            if current is None or chunk.distance < current.distance:
                merged[key] = chunk
        return list(merged.values())

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        parts = []
        for idx, chunk in enumerate(chunks, 1):
            excerpt_id = f"E{idx}"
            parts.append(
                f"--- {excerpt_id} ---\n"
                f"Citation: {chunk.citation}\n"
                f"Distance: {chunk.distance:.4f}\n"
                f"Text:\n{chunk.text[:2200]}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _parse_score_response(response: str) -> Dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start < 0:
                raise ValueError(f"Could not parse rubric score JSON: {response}")
            decoder = json.JSONDecoder()
            try:
                data, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse rubric score JSON: {response}") from exc

        if "score" not in data:
            raise ValueError(f"Rubric score response missing score: {response}")
        return data


def export_scores(
    results: Dict[str, Dict[str, RubricScore]],
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    topics = list(TOPICS.keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["county"]
        + [f"{topic}_score" for topic in topics]
        + [f"{topic}_confidence" for topic in topics]
        + ["composite_score", "composite_percent", "policy_strength_tier"]
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for county in sorted(results.keys()):
            row: Dict[str, Any] = {"county": county}
            composite = 0
            for topic in topics:
                score = results[county].get(topic)
                value = score.score if score else 0
                composite += value
                row[f"{topic}_score"] = value
                row[f"{topic}_confidence"] = score.confidence if score else "none"
            row["composite_score"] = composite
            row["composite_percent"] = round((composite / 15) * 100, 2)
            row["policy_strength_tier"] = composite_tier(composite)
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for county in sorted(results.keys()):
            for topic in topics:
                if topic in results[county]:
                    f.write(json.dumps(results[county][topic].to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score county plans using the 0-3 policy rubric")
    parser.add_argument("--counties", nargs="*", help="Specific counties to score")
    parser.add_argument("--topics", nargs="*", choices=list(TOPICS.keys()), help="Specific topics to score")
    parser.add_argument("--top-k", type=int, default=8, help="Retrieved excerpts per county/topic")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "ai_policy_scores_by_county.csv",
        help="Wide CSV output path",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=PROCESSED_DIR / "ai_policy_scores_by_county.jsonl",
        help="Detailed JSONL output path",
    )
    args = parser.parse_args()

    counties = args.counties or get_unique_counties_from_chunks()
    topics = args.topics or list(TOPICS.keys())
    scorer = RAGRubricScorer()
    results: Dict[str, Dict[str, RubricScore]] = {}

    print(f"Scoring {len(counties)} counties across {len(topics)} topics")
    for county in counties:
        results[county] = {}
        for topic in topics:
            score = scorer.score_topic(county, topic, top_k=args.top_k)
            results[county][topic] = score
            print(f"{county} | {topic}: {score.score} ({score.confidence})")

    export_scores(results, args.output, args.jsonl)
    print(f"Wrote CSV: {args.output}")
    print(f"Wrote JSONL: {args.jsonl}")


if __name__ == "__main__":
    main()
