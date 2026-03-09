from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader

@dataclass
class Page:
    doc_id: str
    page_num: int
    text: str

def extract_pages(pdf_path: Path) -> List[Page]:
    reader = PdfReader(str(pdf_path))
    doc_id = pdf_path.name

    pages: List[Page] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split()) # normalize whitespace
        if text.strip():
            pages.append(Page(doc_id=doc_id, page_num=idx, text=text))
    return pages

def ingest_dir(raw_dir: Path) -> List[Page]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    all_pages: List[Page] = []
    for pdf in pdfs:
        all_pages.extend(extract_pages(pdf))
    return all_pages
