"""
Create citation-safe chunks from page-level JSONL.

Input:
data/processed/pages.jsonl

Output:
data/processed/chunks.jsonl
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES_PATH = ROOT / "data" / "processed" / "pages.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

MAX_CHARS = 5000        # ~800-1000 tokens rough
OVERLAP_CHARS = 1000    # overlap for continuity


def flush_chunk(current_key, page_start, last_page, buffer_text, out):
    county, pdf_file = current_key
    out.write(json.dumps({
        "county": county,
        "pdf_file": pdf_file,
        "page_start": page_start,
        "page_end": last_page,
        "text": buffer_text
    }, ensure_ascii=False) + "\n")


def chunk_pages():
    if not PAGES_PATH.exists():
        raise FileNotFoundError(f"Missing pages file: {PAGES_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    current_key = None
    buffer_text = ""
    page_start = None
    last_page = None
    written = 0

    with PAGES_PATH.open("r", encoding="utf-8") as f, OUT_PATH.open("w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            key = (r["county"], r["pdf_file"])

            # new document boundary
            if key != current_key:
                if current_key is not None and buffer_text.strip():
                    flush_chunk(current_key, page_start, last_page, buffer_text, out)
                    written += 1

                current_key = key
                buffer_text = ""
                page_start = None
                last_page = None

            text = (r.get("text") or "").strip()
            if not text:
                continue

            if not buffer_text:
                page_start = r["page"]

            buffer_text += "\n" + text
            last_page = r["page"]

            if len(buffer_text) >= MAX_CHARS:
                flush_chunk(current_key, page_start, last_page, buffer_text, out)
                written += 1

                # overlap tail for continuity
                buffer_text = buffer_text[-OVERLAP_CHARS:]
                page_start = last_page

        # flush last buffer
        if current_key is not None and buffer_text.strip():
            flush_chunk(current_key, page_start, last_page, buffer_text, out)
            written += 1

    print("Done.")
    print("Chunk file:", OUT_PATH)
    print("Total chunks:", written)


if __name__ == "__main__":
    chunk_pages()