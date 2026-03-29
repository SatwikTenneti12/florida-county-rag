"""
Extract all pages from county PDFs into page-level JSONL format.

Output:
data/processed/pages.jsonl
"""

import json
from pathlib import Path
import fitz  # PyMuPDF


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw_pdfs"
OUT_PATH = ROOT / "data" / "processed" / "pages.jsonl"


def extract_pages():
    pdfs = sorted(RAW_DIR.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    empty_pages = 0
    failed = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            county = pdf.parent.name

            try:
                doc = fitz.open(str(pdf))
            except Exception as e:
                print(f"Failed to open {pdf.name}: {e}")
                failed += 1
                continue

            for pno in range(doc.page_count):
                text = doc[pno].get_text("text") or ""
                tl = len(text.strip())

                if tl == 0:
                    empty_pages += 1

                record = {
                    "county": county,
                    "pdf_file": pdf.name,
                    "relative_path": str(pdf.relative_to(RAW_DIR)),
                    "page": pno + 1,
                    "text": text,
                    "text_len": tl,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            doc.close()

    print("Done.")
    print("Pages written:", written)
    print("Empty pages:", empty_pages)
    print("Failed PDFs:", failed)


if __name__ == "__main__":
    extract_pages()