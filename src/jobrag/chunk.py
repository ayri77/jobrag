from typing import List
from dataclasses import dataclass

from jobrag.ingest import Page
from jobrag.settings import SETTINGS

@dataclass
class Chunk:
    doc_id: str
    page_num: int
    chunk_id: int
    text: str

def chunk_pages(
        pages: List[Page],
        chunk_size: int = SETTINGS.CHUNK_SIZE_CHARS, # chars
        chunk_overlap: int = SETTINGS.CHUNK_OVERLAP_CHARS # chars
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for page in pages:
        t = page.text.strip()
        if not t:
            continue

        start = 0
        cid = 0
        while start < len(t):
            end = min(start + chunk_size, len(t))
            chunk_text = t[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        doc_id=page.doc_id,
                        page_num=page.page_num,
                        chunk_id=cid,
                        text=chunk_text,
                    )
                )
                cid += 1
            if end == len(t):
                break
            start = max(0, end - chunk_overlap)
    return chunks