from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
from typing import List

import numpy as np
import faiss

from jobrag.embed import EmbeddingModel
from jobrag.ingest import ingest_dir
from jobrag.chunk import Chunk, chunk_pages
from jobrag.settings import SETTINGS

def build_index(
        raw_dir: Path,
        output_dir: Path,
        model_name: str = SETTINGS.EMBED_MODEL,
) -> None:

    pages = ingest_dir(raw_dir)
    chunks: List[Chunk] = chunk_pages(pages, chunk_size=SETTINGS.CHUNK_SIZE_CHARS, chunk_overlap=SETTINGS.CHUNK_OVERLAP_CHARS)

    embedder = EmbeddingModel(model_name=model_name)
    texts = [c.text for c in chunks]
    vecs = embedder.embed(texts).astype(np.float32)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "faiss.index"))

    # store chunk metadata aligned by vector id
    meta = [asdict(c) for c in chunks]
    with open(str(output_dir / "chunks_meta.jsonl"), "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Build index: {len(chunks)} chunks, dim={dim}")
    print(f"Saved to: {output_dir}")
