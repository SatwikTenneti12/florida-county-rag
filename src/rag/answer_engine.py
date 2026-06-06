"""
RAG Answer Engine - Generate answers with citations from retrieved context.

This is the core "G" (Generation) component of the RAG system.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retriever import VectorRetriever, RetrievedChunk, get_retriever
from utils.llm import get_llm_client, LLMClient


SYSTEM_PROMPT = """You are an expert policy analyst specializing in Florida county comprehensive plans and environmental regulations.

Your task is to answer questions about county policies based ONLY on the provided document excerpts. 

IMPORTANT RULES:
1. Only use information from the provided excerpts - never make up or assume information
2. Always cite your sources using the format [County Name, p.X] for each claim
3. If the excerpts don't contain relevant information, say so clearly
4. Distinguish between:
   - Enforceable policies (SHALL, MUST, REQUIRED)
   - Goals/objectives (general statements of intent)
   - Recommendations (SHOULD, ENCOURAGE, MAY)
5. Be precise and factual - this is for academic research

When analyzing policy language, note:
- "Shall" indicates a mandatory requirement
- "Should" indicates a recommendation
- "May" indicates optional/permissive language
- Goals and objectives may not be directly enforceable"""

DECOMPOSE_SYSTEM = """You help search a vector database of Florida county comprehensive plans.
Given a user question, output 2-4 short search queries (standalone phrases) that would retrieve relevant plan excerpts.
Use policy-relevant keywords. Do not answer the question — only produce search strings.
Return ONLY valid JSON: an array of strings, e.g. ["query one", "query two"]. No markdown."""

DECOMPOSE_USER = """County filter (if any, else null): {county_json}

User question:
{question}"""


@dataclass
class RAGAnswer:
    """Structured answer from the RAG system."""
    question: str
    answer: str
    sources: List[RetrievedChunk]
    county_filter: Optional[str] = None
    confidence: str = "medium"  # low, medium, high
    retrieval_mode: str = "agent"  # "agent" | "single_pass"
    sub_queries: Optional[List[str]] = None  # agent-decomposed search queries (excludes original question)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "county_filter": self.county_filter,
            "confidence": self.confidence,
            "retrieval_mode": self.retrieval_mode,
        }
        if self.sub_queries is not None:
            d["sub_queries"] = self.sub_queries
        return d
    
    def format_with_citations(self) -> str:
        """Format the answer with a sources section."""
        output = [self.answer, "", "---", "SOURCES:"]
        for i, source in enumerate(self.sources, 1):
            output.append(f"  [{i}] {source.citation}")
            snippet = source.text[:200].replace("\n", " ")
            output.append(f"      \"{snippet}...\"")
        return "\n".join(output)


class RAGAnswerEngine:
    """
    Generate answers to policy questions using retrieval-augmented generation.
    
    This combines:
    1. Vector retrieval to find relevant policy excerpts
    2. LLM generation to synthesize an answer with citations
    """
    
    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.retriever = retriever or get_retriever()
        self.llm = llm_client or get_llm_client()
    
    def answer(
        self,
        question: str,
        county: Optional[str] = None,
        top_k: int = 8,
        include_context: bool = False,
    ) -> RAGAnswer:
        """
        Default path: agent-style retrieval (decompose → multi-query search → one grounded answer).
        For the legacy single-vector-search baseline, use answer_single_pass().
        """
        return self.answer_agent(question, county=county, top_k=top_k)

    def answer_single_pass(
        self,
        question: str,
        county: Optional[str] = None,
        top_k: int = 8,
        include_context: bool = False,
    ) -> RAGAnswer:
        """
        One retrieval pass + generation (classic RAG). Use for benchmarks comparing to older baselines.
        """
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            county_filter=county,
        )

        if not chunks:
            return RAGAnswer(
                question=question,
                answer="No relevant policy excerpts were found for this question.",
                sources=[],
                county_filter=county,
                confidence="low",
                retrieval_mode="single_pass",
                sub_queries=None,
            )

        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, county)
        answer_text = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT)
        confidence = self._assess_confidence(chunks)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            county_filter=county,
            confidence=confidence,
            retrieval_mode="single_pass",
            sub_queries=None,
        )

    def answer_agent(
        self,
        question: str,
        county: Optional[str] = None,
        top_k: int = 8,
    ) -> RAGAnswer:
        """
        Agent-style retrieval: LLM proposes search sub-queries, multi-query retrieval, merge, then answer.
        """
        sub_queries = self._decompose_into_search_queries(question, county)

        # Original question + decomposed queries, deduped by normalized text
        all_q: List[str] = [question.strip()]
        for s in sub_queries:
            t = s.strip()
            if not t:
                continue
            if t.lower() == question.strip().lower():
                continue
            all_q.append(t)

        seen: set = set()
        queries: List[str] = []
        for q in all_q:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q)

        per_k = max(5, min(12, top_k + 2))
        chunks = self.retriever.retrieve_multi_query(
            queries=queries,
            top_k_per_query=per_k,
            county_filter=county,
            deduplicate=True,
        )[:top_k]

        if not chunks:
            return RAGAnswer(
                question=question,
                answer="No relevant policy excerpts were found for this question.",
                sources=[],
                county_filter=county,
                confidence="low",
                retrieval_mode="agent",
                sub_queries=sub_queries or None,
            )

        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, county)
        answer_text = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT)
        confidence = self._assess_confidence(chunks)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            county_filter=county,
            confidence=confidence,
            retrieval_mode="agent",
            sub_queries=sub_queries or None,
        )

    async def answer_agent_stream(
        self,
        question: str,
        county: Optional[str] = None,
        top_k: int = 8,
    ):
        """
        Async generator that yields JSON chunks:
        1. {"type": "metadata", "sources": [...], "sub_queries": [...]}
        2. {"type": "token", "content": "..."}
        """
        sub_queries = self._decompose_into_search_queries(question, county)

        all_q: List[str] = [question.strip()]
        for s in sub_queries:
            t = s.strip()
            if t and t.lower() != question.strip().lower():
                all_q.append(t)

        seen: set = set()
        queries: List[str] = []
        for q in all_q:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q)

        per_k = max(5, min(12, top_k + 2))
        
        # Local chromadb retrieval is synchronous, which is fine for local
        chunks = self.retriever.retrieve_multi_query(
            queries=queries,
            top_k_per_query=per_k,
            county_filter=county,
            deduplicate=True,
        )[:top_k]

        confidence = self._assess_confidence(chunks) if chunks else "low"
        
        # Send metadata first
        metadata = {
            "type": "metadata",
            "sources": [s.to_dict() for s in chunks],
            "sub_queries": sub_queries,
            "confidence": confidence
        }
        yield json.dumps(metadata) + "\n"

        if not chunks:
            yield json.dumps({"type": "token", "content": "No relevant policy excerpts were found for this question."}) + "\n"
            return

        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, county)
        
        messages = [{"role": "user", "content": prompt}]
        
        async for token in self.llm.chat_stream(messages, system_prompt=SYSTEM_PROMPT):
            yield json.dumps({"type": "token", "content": token}) + "\n"

    def _decompose_into_search_queries(
        self, question: str, county: Optional[str]
    ) -> List[str]:
        """LLM → JSON array of short search strings; fallback to empty (caller uses original question only)."""
        county_json = json.dumps(county) if county else "null"
        user = DECOMPOSE_USER.format(county_json=county_json, question=question.strip())
        try:
            raw = self.llm.chat(
                [{"role": "user", "content": user}],
                temperature=0.05,
                max_tokens=400,
                system_prompt=DECOMPOSE_SYSTEM,
            )
            parsed = self._parse_subquery_json(raw)
            out: List[str] = []
            for p in parsed:
                t = str(p).strip()
                if t and len(out) < 5:
                    out.append(t)
            return out
        except Exception:
            return []

    @staticmethod
    def _parse_subquery_json(text: str) -> List[str]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, list):
                        return [str(x).strip() for x in data if str(x).strip()]
                except json.JSONDecodeError:
                    pass
        return []
    
    def answer_with_multi_query(
        self,
        question: str,
        query_variations: List[str],
        county: Optional[str] = None,
        top_k: int = 10
    ) -> RAGAnswer:
        """
        Answer using multiple query variations for better recall.
        
        Useful when the question might be phrased differently in the documents.
        """
        all_queries = [question] + query_variations
        chunks = self.retriever.retrieve_multi_query(
            queries=all_queries,
            top_k_per_query=5,
            county_filter=county,
            deduplicate=True
        )[:top_k]
        
        if not chunks:
            return RAGAnswer(
                question=question,
                answer="No relevant policy excerpts were found.",
                sources=[],
                county_filter=county,
                confidence="low",
                retrieval_mode="multi_query",
                sub_queries=None,
            )
        
        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, county)
        answer_text = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT)
        
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            county_filter=county,
            confidence=self._assess_confidence(chunks),
            retrieval_mode="multi_query",
            sub_queries=list(query_variations),
        )
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"--- EXCERPT {i} ---\n"
                f"Source: {chunk.citation}\n"
                f"Content:\n{chunk.text}\n"
            )
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        county: Optional[str] = None
    ) -> str:
        """Build the prompt for the LLM."""
        county_note = f" (Focus on {county})" if county else ""
        
        return f"""Based on the following excerpts from Florida county comprehensive plans, answer this question{county_note}:

QUESTION: {question}

DOCUMENT EXCERPTS:
{context}

Provide a clear, well-cited answer. Include specific policy language where relevant and cite each source using [County Name, p.X] format. If the evidence is limited or unclear, state that explicitly."""
    
    def _assess_confidence(self, chunks: List[RetrievedChunk]) -> str:
        """Assess answer confidence based on retrieval quality."""
        if not chunks:
            return "low"
        
        # Based on distance scores (lower = better match)
        avg_distance = sum(c.distance for c in chunks) / len(chunks)
        best_distance = min(c.distance for c in chunks)
        
        if best_distance < 0.3 and avg_distance < 0.5:
            return "high"
        elif best_distance < 0.5 and avg_distance < 0.7:
            return "medium"
        else:
            return "low"


# Convenience functions

_engine = None

def get_answer_engine() -> RAGAnswerEngine:
    """Get or create the singleton answer engine."""
    global _engine
    if _engine is None:
        _engine = RAGAnswerEngine()
    return _engine


def ask(question: str, county: Optional[str] = None) -> RAGAnswer:
    """
    Quick function to ask a question.
    
    Usage:
        from rag.answer_engine import ask
        result = ask("Does Alachua County have wildlife corridor policies?", county="Alachua County")
        print(result.format_with_citations())
    """
    return get_answer_engine().answer(question, county=county)
