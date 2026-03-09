from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List, Any

import numpy as np
import faiss

from jobrag.embed import EmbeddingModel
from jobrag.settings import SETTINGS

def load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def main():
    project_root = Path(__file__).resolve().parents[2]
    index_dir = project_root / SETTINGS.INDEX_DIR

    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta = load_meta(index_dir / "chunks_meta.jsonl")

    embedder = EmbeddingModel()

    query = "FastAPI, Docker, deploying ML services"
    qv = embedder.embed([query]).astype(np.float32)

    scores, idx = index.search(qv, 5)

    print("Query:", query)
    print("\nTop results:")
    for i, s in zip(idx[0], scores[0]):
        m = meta[i]
        print(f"- score={s:.4f} | {m['doc_id']} p{m['page_num']} chunk#{m['chunk_id']}")
        print(m["text"][:250])


if __name__ == "__main__":
    main()