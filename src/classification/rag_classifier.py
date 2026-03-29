"""
RAG-based topic classifier.

Uses semantic retrieval + LLM judgment to classify whether counties have
policies for each topic. This replaces simple keyword matching with more
nuanced understanding.

Output:
    data/processed/rag_topic_evidence_by_county.csv
"""

import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TOPICS, PROCESSED_DIR, CHUNKS_PATH
from rag.retriever import VectorRetriever, RetrievedChunk, get_retriever
from utils.llm import get_llm_client
from utils.county_normalizer import normalize_county_name


CLASSIFICATION_PROMPT = """You are analyzing Florida county comprehensive plans to determine if they contain enforceable policies on specific environmental topics.

TOPIC: {topic_name}
TOPIC_KEY: {topic_key}
DESCRIPTION: {topic_description}

COUNTY: {county}

Based on the following excerpts from {county}'s comprehensive plan, determine:
1. Does this county have explicit policies related to {topic_name}?
2. Are these policies enforceable (using SHALL/MUST/REQUIRED language) or just goals/recommendations?

DOCUMENT EXCERPTS:
{context}

Respond with a JSON object:
{{
    "has_policy": true/false,
    "confidence": "high"/"medium"/"low",
    "policy_type": "enforceable"/"goal"/"recommendation"/"none",
    "evidence_summary": "Brief summary of the evidence found",
    "key_quotes": ["Direct quote 1", "Direct quote 2"]
}}

IMPORTANT:
- Only say has_policy=true if there is clear evidence in the excerpts that the policy explicitly requires the topic action
- "enforceable" means explicit requirements (shall, must, required)
- "goal" means objectives without enforcement mechanism
- If excerpts do not clearly mention the topic ACTION as a requirement, has_policy must be false

Topic-specific evidence rules:
- wildlife_surveys: Evidence must explicitly require conducting wildlife/species/habitat surveys, inventories, or monitoring (e.g., baseline/preconstruction surveys or "shall conduct" before development). General commitments to protect species, coordinate with agencies, or create conservation maps are NOT sufficient unless they explicitly require a survey/inventory/monitoring process.
- wildlife_crossings: Evidence must explicitly require wildlife crossings or passage infrastructure (e.g., underpass, overpass, ecopassage/fauna passage, bridge/culvert) or require passage through barriers as a condition. General road safety text or non-specific "crossings" references are NOT sufficient unless the policy clearly addresses wildlife crossing/passage infrastructure.

Use the key_quotes to show the exact lines that contain both (a) the topic action and (b) enforceability language."""


# Guardrails: before calling the LLM, only keep chunks that contain
# (1) enforceable policy language and (2) topic-specific "action" terms.
# This reduces false positives caused by loosely related excerpts.
ENFORCEABLE_RE = re.compile(r"\b(shall|must|required|require[s]?)\b", re.IGNORECASE)

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


def passes_guardrail(topic_key: str, text: str) -> bool:
    """
    Guardrail filter to reduce topic misclassification.

    For wildlife_surveys and wildlife_crossings we require topic-action terms
    (survey/inventory/monitoring or crossing infrastructure). We do NOT require
    enforceability text here to avoid killing recall; the LLM prompt still
    enforces the "enforceable policy" definition.
    """
    if not text:
        return False
    action_re = ACTION_PATTERNS.get(topic_key)
    if not action_re:
        return False
    if topic_key in ("wildlife_surveys", "wildlife_crossings"):
        return action_re.search(text) is not None

    # For other topics, don't block anything (guardrail is only strict for
    # the hard/ambiguous topics).
    return True


@dataclass
class TopicClassification:
    """Classification result for a single topic in a county."""
    county: str
    topic: str
    has_policy: bool
    confidence: str
    policy_type: str
    evidence_summary: str
    key_quotes: List[str]
    source_chunks: List[RetrievedChunk]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "county": self.county,
            "topic": self.topic,
            "has_policy": self.has_policy,
            "confidence": self.confidence,
            "policy_type": self.policy_type,
            "evidence_summary": self.evidence_summary,
            "key_quotes": self.key_quotes,
            "sources": [c.citation for c in self.source_chunks]
        }


class RAGTopicClassifier:
    """
    Classify counties by topic using RAG.
    
    For each county-topic pair:
    1. Retrieve relevant chunks using topic-specific queries
    2. Use LLM to judge if the county has enforceable policies
    3. Return structured classification with evidence
    """
    
    def __init__(self, retriever: Optional[VectorRetriever] = None):
        self.retriever = retriever or get_retriever()
        self.llm = get_llm_client()
    
    def classify_topic(
        self,
        county: str,
        topic_key: str,
        top_k: int = 3
    ) -> TopicClassification:
        """
        Classify a single county-topic pair.
        
        Args:
            county: County name (as it appears in the vector DB)
            topic_key: Topic key from TOPICS config
            top_k: Number of chunks to retrieve
            
        Returns:
            TopicClassification with evidence and judgment
        """
        topic_config = TOPICS[topic_key]
        
        # Retrieve candidate chunks (high recall), then rerank deterministically
        # before sending only the best ones to the LLM.
        queries = [topic_config["query"]] + topic_config["keywords"][:2]
        candidate_per_query = max(10, top_k * 6)

        chunks = self.retriever.retrieve_multi_query(
            queries=queries,
            top_k_per_query=candidate_per_query,
            county_filter=county,
            deduplicate=True,
        )
        # Cap for speed while keeping recall
        chunks = chunks[: min(len(chunks), candidate_per_query * max(1, len(queries)))]
        
        if not chunks:
            return TopicClassification(
                county=county,
                topic=topic_key,
                has_policy=False,
                confidence="high",
                policy_type="none",
                evidence_summary="No relevant excerpts found for this county.",
                key_quotes=[],
                source_chunks=[]
            )

        # If the topic is one of the ambiguous ones, ensure at least one
        # candidate chunk contains topic-action terms; otherwise skip the LLM.
        action_re = ACTION_PATTERNS.get(topic_key)
        if topic_key in ("wildlife_surveys", "wildlife_crossings") and action_re:
            if not any(action_re.search(c.text) for c in chunks):
                return TopicClassification(
                    county=county,
                    topic=topic_key,
                    has_policy=False,
                    confidence="high",
                    policy_type="none",
                    evidence_summary="No topic-action survey/crossing language found in retrieved excerpts.",
                    key_quotes=[],
                    source_chunks=[],
                )

        def score_chunk(c: RetrievedChunk) -> float:
            t = (c.text or "").lower()
            score = 0.0

            # Reward action/topic signal
            if action_re and action_re.search(c.text):
                score += 6.0

            # Reward enforceable policy language
            if ENFORCEABLE_RE.search(t):
                score += 2.0

            # Penalize common ambiguous/irrelevant contexts
            if topic_key == "wildlife_crossings":
                if re.search(r"\b(pedestrian|crosswalk|crossing guard|road crossing|railroad crossing|train crossing)\b", t):
                    score -= 4.0

            return score

        ranked = sorted(chunks, key=score_chunk, reverse=True)
        chunks = ranked[:top_k]

        # Build context and prompt
        context = self._build_context(chunks)
        prompt = CLASSIFICATION_PROMPT.format(
            topic_name=topic_config["display_name"],
            topic_key=topic_key,
            topic_description=topic_config["description"],
            county=county,
            context=context
        )
        
        # Get LLM judgment
        try:
            response = self.llm.complete(prompt, temperature=0.0)
            result = self._parse_response(response)
        except Exception as e:
            # Fallback to keyword-based classification
            has_keywords = self._check_keywords(chunks, topic_config["keywords"])
            result = {
                "has_policy": has_keywords,
                "confidence": "low",
                "policy_type": "unknown",
                "evidence_summary": f"LLM classification failed: {e}. Fell back to keyword matching.",
                "key_quotes": []
            }
        
        return TopicClassification(
            county=county,
            topic=topic_key,
            has_policy=result.get("has_policy", False),
            confidence=result.get("confidence", "low"),
            policy_type=result.get("policy_type", "unknown"),
            evidence_summary=result.get("evidence_summary", ""),
            key_quotes=result.get("key_quotes", []),
            source_chunks=chunks
        )
    
    def classify_county(
        self,
        county: str,
        topics: Optional[List[str]] = None,
        top_k_chunks: int = 3,
    ) -> Dict[str, TopicClassification]:
        """Classify all topics for a single county."""
        if topics is None:
            topics = list(TOPICS.keys())
        
        results = {}
        for topic in topics:
            results[topic] = self.classify_topic(county, topic, top_k=top_k_chunks)
        
        return results
    
    def classify_all_counties(
        self,
        counties: List[str],
        topics: Optional[List[str]] = None,
        progress_callback=None,
        top_k_chunks: int = 3,
    ) -> Dict[str, Dict[str, TopicClassification]]:
        """
        Classify all topics for multiple counties.
        
        Args:
            counties: List of county names
            topics: List of topic keys (default: all)
            progress_callback: Optional callback(county, topic, result)
            
        Returns:
            Nested dict: results[county][topic] = TopicClassification
        """
        if topics is None:
            topics = list(TOPICS.keys())
        
        all_results = {}
        
        for county in counties:
            all_results[county] = {}
            for topic in topics:
                result = self.classify_topic(county, topic, top_k=top_k_chunks)
                all_results[county][topic] = result
                
                if progress_callback:
                    progress_callback(county, topic, result)
        
        return all_results
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"--- Excerpt {i} ({chunk.pdf_file}, pages {chunk.page_start}-{chunk.page_end}) ---\n"
                f"{chunk.text[:1200]}\n"
            )
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            import re
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            
            # Return default
            return {
                "has_policy": False,
                "confidence": "low",
                "policy_type": "unknown",
                "evidence_summary": "Failed to parse LLM response",
                "key_quotes": []
            }
    
    def _check_keywords(self, chunks: List[RetrievedChunk], keywords: List[str]) -> bool:
        """Fallback keyword-based classification."""
        combined_text = " ".join(c.text.lower() for c in chunks)
        return any(kw.lower() in combined_text for kw in keywords)


def get_unique_counties_from_chunks() -> List[str]:
    """Get list of unique counties from the chunks file."""
    counties = set()
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counties.add(r["county"])
    return sorted(counties)


def export_results_csv(
    results: Dict[str, Dict[str, TopicClassification]],
    output_path: Path
):
    """Export classification results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    topics = list(TOPICS.keys())
    fieldnames = ["county"] + topics + [f"{t}_confidence" for t in topics]
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for county in sorted(results.keys()):
            row = {"county": county}
            for topic in topics:
                classification = results[county].get(topic)
                if classification:
                    row[topic] = classification.has_policy
                    row[f"{topic}_confidence"] = classification.confidence
                else:
                    row[topic] = False
                    row[f"{topic}_confidence"] = "none"
            writer.writerow(row)
    
    print(f"Wrote results to {output_path}")


def export_detailed_results(
    results: Dict[str, Dict[str, TopicClassification]],
    output_path: Path
):
    """Export detailed results with evidence as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for county in sorted(results.keys()):
            for topic, classification in results[county].items():
                record = classification.to_dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify counties by environmental policy topics using RAG"
    )
    parser.add_argument(
        "--counties",
        type=str,
        nargs="*",
        help="Specific counties to classify (default: all)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="*",
        choices=list(TOPICS.keys()),
        help="Specific topics to classify (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DIR / "rag_topic_evidence_by_county.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also output detailed JSONL with evidence"
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        default=3,
        help="How many retrieved chunks to send to the LLM per county-topic"
    )
    
    args = parser.parse_args()
    
    # Get counties
    if args.counties:
        counties = args.counties
    else:
        counties = get_unique_counties_from_chunks()
    
    print(f"Classifying {len(counties)} counties")
    print(f"Topics: {args.topics or list(TOPICS.keys())}")
    
    classifier = RAGTopicClassifier()
    
    def progress(county, topic, result):
        status = "YES" if result.has_policy else "NO"
        print(f"  {county} | {topic}: {status} ({result.confidence})")
    
    results = classifier.classify_all_counties(
        counties=counties,
        topics=args.topics,
        top_k_chunks=args.top_k_chunks,
        progress_callback=progress
    )
    
    # Export results
    output_path = Path(args.output)
    export_results_csv(results, output_path)
    
    if args.detailed:
        detailed_path = output_path.with_suffix(".jsonl")
        export_detailed_results(results, detailed_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
