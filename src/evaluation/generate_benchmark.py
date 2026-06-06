import json
import csv
import logging
import sys
import os
from pathlib import Path

# Ensure src is in python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rag.answer_engine import get_answer_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "benchmarks" / "rag_baseline_questions.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "benchmark_output.csv"

def generate_benchmark_data():
    if not INPUT_FILE.exists():
        logger.error(f"Cannot find input file: {INPUT_FILE}")
        return

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    engine = get_answer_engine()
    
    logger.info("Starting benchmark generation...")
    
    # Read questions
    questions = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Open CSV for writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['question', 'context', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for q_obj in questions:
            question_text = q_obj.get("question")
            county = q_obj.get("county")
            
            logger.info(f"Processing Q: {question_text[:50]}...")
            
            try:
                # Use the synchronous answer agent for batch processing
                result = engine.answer_agent(
                    question=question_text,
                    county=county,
                    top_k=5  # Adjust as needed
                )
                
                # Combine source excerpts into a single context string
                sources = result.sources
                context_chunks = [s.text for s in sources]
                combined_context = "\n\n".join(context_chunks)
                
                # Write row
                writer.writerow({
                    'question': question_text,
                    'context': combined_context,
                    'answer': result.answer
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {question_text} - {e}")
                
    logger.info(f"Successfully generated benchmark data at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_benchmark_data()
