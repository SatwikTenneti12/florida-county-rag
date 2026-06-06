import asyncio
import csv
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.llm import get_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROCESSED_DIR = ROOT / "data" / "processed"

# Evaluation Prompts tailored to Florida County Comprehensive Plans
GROUNDEDNESS_PROMPT = """You are a Legal Policy Auditor evaluating an AI's answer against official Florida County Comprehensive Plan documents.

[Retrieved Context]
{context}

[AI Generated Answer]
{answer}

Task: Evaluate if the AI Generated Answer is strictly grounded in the Retrieved Context.
Score from 0 to 5, where:
0 = Completely hallucinated or ignores context.
5 = 100% of claims are directly supported by the context.

Provide the output strictly in JSON format:
{{
  "score": <int>,
  "justification": "<string>"
}}"""

CONTEXT_RELEVANCE_PROMPT = """You are an expert in Information Retrieval evaluating search results.

[User Question]
{question}

[Retrieved Context]
{context}

Task: Evaluate how relevant the Retrieved Context is to answering the User Question.
Score from 0 to 5, where:
0 = Completely irrelevant policy chunks.
5 = Contains the exact policy text needed to answer the question perfectly.

Provide the output strictly in JSON format:
{{
  "score": <int>,
  "justification": "<string>"
}}"""

ANSWER_RELEVANCE_PROMPT = """You are a Quality Assurance tester evaluating an AI assistant.

[User Question]
{question}

[AI Generated Answer]
{answer}

Task: Evaluate how well the AI Generated Answer actually answers the User Question.
Score from 0 to 5, where:
0 = Answer does not address the question at all.
5 = Answer perfectly addresses the question.

Provide the output strictly in JSON format:
{{
  "score": <int>,
  "justification": "<string>"
}}"""

def parse_judge_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def judge_json(llm, prompt: str, metric_name: str) -> dict:
    repair_prompt = None
    last_error = None

    for attempt in range(2):
        try:
            content = repair_prompt or prompt
            resp = await llm.chat_async(
                [{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=700,
                system_prompt="Return only compact valid JSON. Do not include markdown.",
            )
            data = parse_judge_json(resp)
            score = int(data.get("score", 0))
            data["score"] = max(0, min(5, score))
            return data
        except Exception as e:
            last_error = e
            repair_prompt = (
                f"The previous {metric_name} judge response was not valid JSON. "
                "Re-evaluate the same task and return only this exact JSON shape: "
                '{"score": 0, "justification": "short reason"}\n\n'
                f"TASK:\n{prompt}"
            )

    logger.error(f"{metric_name} eval failed after retry: {last_error}")
    return {"score": 0, "justification": f"{metric_name} evaluation failed"}

async def run_evaluation(csv_file: Path):
    """
    Reads a CSV file containing columns: 'question', 'context', 'answer'.
    Runs the LLM-as-a-judge for Groundedness, Context Relevance, and Answer Relevance.
    Outputs the final metrics.
    """
    if not csv_file.exists():
        logger.error(f"Cannot find dataset at {csv_file}")
        return

    results = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        llm = get_llm_client()
        for row in reader:
            q = row.get('question', '')
            ctx = row.get('context', '')
            ans = row.get('answer', '')
            
            g_data = await judge_json(
                llm,
                GROUNDEDNESS_PROMPT.format(context=ctx, answer=ans),
                "Groundedness",
            )
            cr_data = await judge_json(
                llm,
                CONTEXT_RELEVANCE_PROMPT.format(question=q, context=ctx),
                "Context Relevance",
            )
            ar_data = await judge_json(
                llm,
                ANSWER_RELEVANCE_PROMPT.format(question=q, answer=ans),
                "Answer Relevance",
            )
                
            results.append({
                "question": q,
                "groundedness": g_data.get("score", 0),
                "context_relevance": cr_data.get("score", 0),
                "answer_relevance": ar_data.get("score", 0),
                "groundedness_justification": g_data.get("justification", ""),
                "context_relevance_justification": cr_data.get("justification", ""),
                "answer_relevance_justification": ar_data.get("justification", ""),
            })
            logger.info(f"Evaluated Q: {q[:30]}... | Scores: G={g_data.get('score')} CR={cr_data.get('score')} AR={ar_data.get('score')}")

    if not results:
        return
        
    avg_g = sum(r["groundedness"] for r in results) / len(results)
    avg_cr = sum(r["context_relevance"] for r in results) / len(results)
    avg_ar = sum(r["answer_relevance"] for r in results) / len(results)
    
    final_report = {
        "total_evaluated": len(results),
        "average_groundedness_5": avg_g,
        "average_context_relevance_5": avg_cr,
        "average_answer_relevance_5": avg_ar,
        "details": results
    }
    
    out_path = PROCESSED_DIR / "validation_results.json"
    with open(out_path, 'w') as f:
        json.dump(final_report, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {out_path}")
    logger.info(f"Avg Groundedness: {avg_g:.2f}/5")
    logger.info(f"Avg Context Relevance: {avg_cr:.2f}/5")
    logger.info(f"Avg Answer Relevance: {avg_ar:.2f}/5")

if __name__ == "__main__":
    asyncio.run(run_evaluation(PROCESSED_DIR / "benchmark_output.csv"))
