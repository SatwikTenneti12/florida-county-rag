"""
RAG Answer Engine - Generate answers with citations from retrieved context.

This is the core "G" (Generation) component of the RAG system.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

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


@dataclass
class RAGAnswer:
    """Structured answer from the RAG system."""
    question: str
    answer: str
    sources: List[RetrievedChunk]
    county_filter: Optional[str] = None
    confidence: str = "medium"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "county_filter": self.county_filter,
            "confidence": self.confidence
        }
    
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
        include_context: bool = False
    ) -> RAGAnswer:
        """
        Answer a question about county policies.
        
        Args:
            question: The policy question to answer
            county: Optional county to focus on
            top_k: Number of chunks to retrieve
            include_context: Whether to include raw context in response
            
        Returns:
            RAGAnswer with the generated answer and sources
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            county_filter=county
        )
        
        if not chunks:
            return RAGAnswer(
                question=question,
                answer="No relevant policy excerpts were found for this question.",
                sources=[],
                county_filter=county,
                confidence="low"
            )
        
        # Build context for LLM
        context = self._build_context(chunks)
        
        # Generate answer
        prompt = self._build_prompt(question, context, county)
        answer_text = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT)
        
        # Determine confidence based on retrieval quality
        confidence = self._assess_confidence(chunks)
        
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            county_filter=county,
            confidence=confidence
        )
    
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
                confidence="low"
            )
        
        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, county)
        answer_text = self.llm.complete(prompt, system_prompt=SYSTEM_PROMPT)
        
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            county_filter=county,
            confidence=self._assess_confidence(chunks)
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
