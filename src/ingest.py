"""
PDF ingestion pipeline.

Usage: python src/ingest.py --corpus data/corpus/ --out data/chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import pymupdf4llm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 80

_MODEL_RE = re.compile(r"\b([A-Z]{1,4}[-]?[A-Z0-9]{2,8}(?:[-][A-Z0-9]{2,8})?)\b")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)

def _extract_model_numbers(text: str) -> list[str]:
    seen, result = set(), []
    for c in _MODEL_RE.findall(text):
        if c not in seen and re.search(r"\d", c) and len(c) >= 3:
            seen.add(c)
            result.append(c)
    return result[:20]

def _is_table_chunk(text: str) -> bool:
    return len(_TABLE_LINE_RE.findall(text)) >= 2

def _split_long_section(title: str, text: str, page: int) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf, buf_len = [], [], 0
    for para in paragraphs:
        if buf_len + len(para) > MAX_CHUNK_CHARS and buf:
            chunks.append({"title": title, "text": "\n\n".join(buf), "page": page})
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para)
    if buf:
        chunks.append({"title": title, "text": "\n\n".join(buf), "page": page})
    return chunks

def _page_to_sections(page_text: str, page_num: int, current_title: str):
    sections = []
    headings = list(_HEADING_RE.finditer(page_text))

    if not headings:
        text = page_text.strip()
        if len(text) >= MIN_CHUNK_CHARS:
            sections.append({"title": current_title, "text": text, "page": page_num})
        return sections, current_title

    pre = page_text[:headings[0].start()].strip()
    if len(pre) >= MIN_CHUNK_CHARS:
        sections.append({"title": current_title, "text": pre, "page": page_num})

    for i, m in enumerate(headings):
        title = m.group(2).strip().strip("*").strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(page_text)
        body = page_text[start:end].strip()
        if len(body) >= MIN_CHUNK_CHARS:
            sections.append({"title": title, "text": body, "page": page_num})
        current_title = title

    return sections, current_title

def process_pdf(pdf_path: Path) -> list[dict]:
    doc_id = pdf_path.stem
    logger.info(f"Processing {doc_id} ...")

    try:
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    except Exception as e:
        logger.error(f"  Failed to parse {pdf_path}: {e}")
        return []

    header_text = " ".join(p["text"] for p in pages[:3])
    model_numbers = _extract_model_numbers(header_text)
    logger.info(f"  {len(pages)} pages  models={model_numbers[:5]}")

    raw_sections: list[dict] = []
    current_title = "Document"

    for page_data in pages:
        page_num = page_data["metadata"]["page_number"]
        text = page_data["text"]
        if not text.strip():
            continue
        secs, current_title = _page_to_sections(text, page_num, current_title)
        raw_sections.extend(secs)

    chunks, idx = [], 0
    for sec in raw_sections:
        title, text, page = sec["title"], sec["text"], sec["page"]
        full_text = f"{title}\n\n{text}" if title != "Document" else text

        pieces = (
            [{"title": title, "text": full_text, "page": page}]
            if len(full_text) <= MAX_CHUNK_CHARS
            else _split_long_section(title, full_text, page)
        )

        for piece in pieces:
            chunks.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "page": piece["page"],
                "section_title": piece["title"],
                "chunk_type": "table" if _is_table_chunk(piece["text"]) else "text",
                "text": piece["text"],
                "model_numbers": model_numbers,
                "char_count": len(piece["text"]),
            })
            idx += 1

    logger.info(f"  → {len(chunks)} chunks")
    return chunks

def ingest(corpus_dir: Path, out_path: Path) -> list[dict]:
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {corpus_dir}")
        return []

    logger.info(f"Found {len(pdfs)} PDFs")
    all_chunks: list[dict] = []
    for pdf in pdfs:
        all_chunks.extend(process_pdf(pdf))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_chunks, indent=2))
    logger.info(f"Saved {len(all_chunks)} chunks → {out_path}")
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/"))
    parser.add_argument("--out", type=Path, default=Path("data/chunks.json"))
    args = parser.parse_args()
    ingest(args.corpus, args.out)
