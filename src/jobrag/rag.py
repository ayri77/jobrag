from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss

from jobrag.embed import EmbeddingModel
from jobrag.llm import OllamaClient
from jobrag.settings import SETTINGS

import time
import logging

MAX_CHUNK_CHARS = SETTINGS.MAX_CHUNK_CHARS_IN_PROMPT
logger = logging.getLogger("jobrag")


def load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


@dataclass
class RAGEngine:
    embedder: EmbeddingModel
    llm: OllamaClient
    index: faiss.Index
    meta: List[Dict[str, Any]]

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        embed_model: str = SETTINGS.EMBED_MODEL,
        ollama_host: str = SETTINGS.OLLAMA_HOST,
        ollama_model: str = SETTINGS.OLLAMA_MODEL,
    ) -> "RAGEngine":

        index = faiss.read_index(str(index_dir / "faiss.index"))
        meta = load_meta(index_dir / "chunks_meta.jsonl")
        embedder = EmbeddingModel(model_name=embed_model)
        llm = OllamaClient(host=ollama_host, model=ollama_model)
        return cls(embedder=embedder, llm=llm, index=index, meta=meta)

    def retrieve(self, query: str, top_k: int = SETTINGS.TOP_K_QA) -> List[Dict[str, Any]]:
        qv = self.embedder.embed([query]).astype(np.float32)
        scores, idx = self.index.search(qv, top_k)

        results = []
        for i, s in zip(idx[0], scores[0]):
            m = dict(self.meta[i])
            m["score"] = float(s)
            results.append(m)
        return results

    def answer(self, query: str, top_k: int = SETTINGS.TOP_K_QA, stream: bool = False) -> Tuple[str, List[Dict[str, Any]]]:

        t0 = time.perf_counter()
        docs = self.retrieve(query, top_k=top_k)
        t1 = time.perf_counter()

        context_blocks = []
        for d in docs:
            ref = f"{d['doc_id']} p{d['page_num']}"
            context_blocks.append(f"[{ref}]\n{d['text'][:MAX_CHUNK_CHARS]}\n")

        context = "\n".join(context_blocks)

        system = (
            "You are a job search assistant. Use ONLY the provided context from the user's documents. "
            "If the context does not contain the answer, say: 'Not found in documents.' "
            "Do not invent facts. Keep answers concise and professional. "
            "Citations are mandatory. Use ONLY citations from the SOURCES list. "
            "Citation format must be exactly: [<doc_id> p<page_num>]. "
            "Do NOT use generic citations like [Document pX]."
        )

        sources = "\n".join([f"- {d['doc_id']} p{d['page_num']}" for d in docs])

        prompt = (
            f"QUESTION:\n{query}\n\n"
            f"SOURCES (allowed citations):\n{sources}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer using only the context.\n"
            f"- If not found, say: Not found in documents.\n"
            f"- Add sources at the end using ONLY the allowed citations.\n"
            f"- After the sources, output the exact token: END_OF_ANSWER\n"
        )

        if stream:
            out = self.llm.generate_stream(prompt=prompt, system=system, temperature=0.0)
        else:
            out = self.llm.generate(prompt=prompt, system=system, temperature=0.0)

        t2 = time.perf_counter()

        if SETTINGS.DEBUG:
            logger.debug("[DEBUG] ollama keys:", sorted(out.keys()))
            logger.debug("[DEBUG] response len:", len(out.get("response", "")))
            logger.debug("[DEBUG] done_reason:", out.get("done_reason"))
            logger.debug("[DEBUG] done:", out.get("done"))
            logger.debug("[DEBUG] response preview:", repr(out.get("response", "")[:200]))

        if SETTINGS.LOG_TIMING:
            logger.debug(f"[TIMING] retrieve: {t1 - t0:.2f}s | generate: {t2 - t1:.2f}s | total: {t2 - t0:.2f}s")

        return out["response"].strip().replace("<", "").replace(">", ""), docs

    def summarize_for_jd(
        self,
        job_description: str,
        top_k: int = SETTINGS.TOP_K_JD,
        stream: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:

        t0 = time.perf_counter()
        docs = self.retrieve(job_description, top_k=top_k)
        t1 = time.perf_counter()

        context_blocks = []
        for d in docs:
            ref = f"{d['doc_id']} p{d['page_num']}"
            context_blocks.append(f"[{ref}]\n{d['text']}\n")
        context = "\n".join(context_blocks)

        system = (
            "You are a professional resume writer. Use ONLY the provided context from the candidate's documents. "
            "Do not invent facts. If something is not supported by the context, omit it. "
            "Write concise, role-relevant English. "
            "Citations are mandatory. Use ONLY citations from the SOURCES list. "
            "Citation format must be exactly: [<doc_id> p<page_num>]. "
            "Do NOT use generic citations like [Document pX]."
        )

        sources = "\n".join([f"- {d['doc_id']} p{d['page_num']}" for d in docs])

        prompt = (
            "JOB DESCRIPTION:\n"
            f"{job_description}\n\n"
            f"SOURCES (allowed citations):\n{sources}\n\n"
            "CANDIDATE CONTEXT:\n"
            f"{context}\n\n"
            "TASK:\n"
            "- Write a concise summary of the candidate's relevant experience for this JD.\n"
            "- Focus on the most relevant skills, projects, and outcomes.\n"
            "- Use 3–4 bullet points max.\n"
            "- Each bullet must end with citations from SOURCES.\n"
            "- Keep it under 120 words.\n"
        )

        if stream:
            out = self.llm.generate_stream(prompt=prompt, system=system, temperature=0.0)
        else:
            out = self.llm.generate(prompt=prompt, system=system, temperature=0.0)

        t2 = time.perf_counter()

        if SETTINGS.DEBUG:
            logger.debug("[DEBUG] ollama keys: %s", sorted(out.keys()))
            logger.debug("[DEBUG] response len: %s", len(out.get("response", "")))
            logger.debug("[DEBUG] done_reason: %s", out.get("done_reason"))
            logger.debug("[DEBUG] done: %s", out.get("done"))
            logger.debug("[DEBUG] response preview: %s", repr(out.get("response", "")[:200]))

        if SETTINGS.LOG_TIMING:
            logger.debug(f"[TIMING] retrieve: {t1 - t0:.2f}s | generate: {t2 - t1:.2f}s | total: {t2 - t0:.2f}s")

        return out["response"].strip().replace("<", "").replace(">", ""), docs