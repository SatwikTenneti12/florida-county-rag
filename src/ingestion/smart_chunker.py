"""
Smart chunking with sentence-aware splitting.

Improvements over basic chunking:
1. Respects sentence boundaries to avoid cutting mid-sentence
2. Attempts to keep policy sections together
3. Better metadata tracking
4. Configurable overlap strategies

Input:
    data/processed/pages.jsonl

Output:
    data/processed/chunks_v2.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PAGES_PATH, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """A text chunk with citation metadata."""
    county: str
    pdf_file: str
    page_start: int
    page_end: int
    text: str
    chunk_index: int  # Index within the document
    
    def to_dict(self) -> dict:
        return {
            "county": self.county,
            "pdf_file": self.pdf_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "chunk_index": self.chunk_index
        }


class SmartChunker:
    """
    Sentence-aware text chunker that preserves citations.
    
    Features:
    - Splits on sentence boundaries when possible
    - Keeps policy statements together
    - Maintains overlap for context continuity
    - Tracks page ranges for citations
    """
    
    # Patterns that indicate policy section boundaries
    SECTION_PATTERNS = [
        r'^POLICY\s+\d+',
        r'^OBJECTIVE\s+\d+',
        r'^GOAL\s+\d+',
        r'^\d+\.\d+\s+[A-Z]',  # Numbered sections like "1.2 Wildlife"
        r'^[A-Z][A-Z\s]+:',   # ALL CAPS headers
    ]
    
    def __init__(
        self,
        max_chars: int = CHUNK_SIZE,
        overlap_chars: int = CHUNK_OVERLAP,
        min_chunk_chars: int = MIN_CHUNK_SIZE
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        
        # Compile patterns
        self.section_re = re.compile(
            '|'.join(f'({p})' for p in self.SECTION_PATTERNS),
            re.MULTILINE
        )
    
    def chunk_document(
        self,
        pages: List[Tuple[int, str]],
        county: str,
        pdf_file: str
    ) -> List[Chunk]:
        """
        Chunk a document given its pages.
        
        Args:
            pages: List of (page_number, text) tuples
            county: County name
            pdf_file: PDF filename
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        buffer_text = ""
        buffer_pages = []
        chunk_index = 0
        
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            
            # Add page to buffer
            buffer_text += "\n" + page_text
            buffer_pages.append(page_num)
            
            # Check if we should flush
            while len(buffer_text) >= self.max_chars:
                # Find a good split point
                split_point = self._find_split_point(buffer_text, self.max_chars)
                
                chunk_text = buffer_text[:split_point].strip()
                if len(chunk_text) >= self.min_chunk_chars:
                    chunks.append(Chunk(
                        county=county,
                        pdf_file=pdf_file,
                        page_start=buffer_pages[0],
                        page_end=buffer_pages[-1],
                        text=chunk_text,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                
                # Keep overlap
                overlap_start = max(0, split_point - self.overlap_chars)
                buffer_text = buffer_text[overlap_start:]
                
                # Update page tracking (approximate)
                if len(buffer_pages) > 1:
                    buffer_pages = buffer_pages[-2:]  # Keep last 2 pages for overlap
        
        # Flush remaining buffer
        if buffer_text.strip() and len(buffer_text.strip()) >= self.min_chunk_chars:
            chunks.append(Chunk(
                county=county,
                pdf_file=pdf_file,
                page_start=buffer_pages[0] if buffer_pages else 1,
                page_end=buffer_pages[-1] if buffer_pages else 1,
                text=buffer_text.strip(),
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _find_split_point(self, text: str, target: int) -> int:
        """
        Find the best point to split text near the target position.
        
        Priority:
        1. Section boundary (POLICY, OBJECTIVE, etc.)
        2. Paragraph boundary (double newline)
        3. Sentence boundary (. ! ?)
        4. Word boundary (space)
        5. Target position (fallback)
        """
        # Search window: 20% before target to target
        window_start = int(target * 0.8)
        window_end = min(target + 200, len(text))  # Small buffer past target
        
        search_text = text[window_start:window_end]
        
        # 1. Look for section boundaries
        match = self.section_re.search(search_text)
        if match and match.start() > 50:  # Not at very start of window
            return window_start + match.start()
        
        # 2. Look for paragraph boundaries
        para_match = re.search(r'\n\s*\n', search_text)
        if para_match:
            return window_start + para_match.end()
        
        # 3. Look for sentence boundaries
        sent_match = re.search(r'[.!?]\s+(?=[A-Z])', search_text)
        if sent_match:
            return window_start + sent_match.end()
        
        # 4. Look for any sentence end
        sent_match = re.search(r'[.!?]\s+', search_text)
        if sent_match:
            return window_start + sent_match.end()
        
        # 5. Word boundary
        space_idx = search_text.rfind(' ')
        if space_idx > 0:
            return window_start + space_idx + 1
        
        # 6. Fallback to target
        return target
    
    def process_pages_file(
        self,
        pages_path: Path = PAGES_PATH,
        output_path: Optional[Path] = None
    ) -> int:
        """
        Process the pages.jsonl file and create chunks.
        
        Returns number of chunks written.
        """
        if output_path is None:
            output_path = PROCESSED_DIR / "chunks_v2.jsonl"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        current_doc = None
        pages_buffer = []
        written = 0
        
        with pages_path.open("r", encoding="utf-8") as f_in, \
             output_path.open("w", encoding="utf-8") as f_out:
            
            for line in f_in:
                record = json.loads(line)
                doc_key = (record["county"], record["pdf_file"])
                
                # New document
                if doc_key != current_doc:
                    # Process previous document
                    if current_doc and pages_buffer:
                        chunks = self.chunk_document(
                            pages_buffer,
                            current_doc[0],
                            current_doc[1]
                        )
                        for chunk in chunks:
                            f_out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                            written += 1
                    
                    current_doc = doc_key
                    pages_buffer = []
                
                pages_buffer.append((record["page"], record.get("text", "")))
            
            # Process last document
            if current_doc and pages_buffer:
                chunks = self.chunk_document(
                    pages_buffer,
                    current_doc[0],
                    current_doc[1]
                )
                for chunk in chunks:
                    f_out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                    written += 1
        
        return written


def main():
    """Re-chunk the pages with smart splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart chunking with sentence awareness")
    parser.add_argument("--max-chars", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "chunks_v2.jsonl"))
    
    args = parser.parse_args()
    
    chunker = SmartChunker(
        max_chars=args.max_chars,
        overlap_chars=args.overlap
    )
    
    print(f"Processing {PAGES_PATH}")
    print(f"Max chunk size: {args.max_chars} chars")
    print(f"Overlap: {args.overlap} chars")
    
    written = chunker.process_pages_file(output_path=Path(args.output))
    
    print(f"\nDone! Wrote {written} chunks to {args.output}")


if __name__ == "__main__":
    main()
