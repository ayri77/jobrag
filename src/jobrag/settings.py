from __future__ import annotations
from dataclasses import dataclass

import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Paths (relative to project root; tests resolve root separately)
    RAW_DIR: str = "data/raw"
    INDEX_DIR: str = "data/index"

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-m3"

    # Chunking
    CHUNK_SIZE_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 200

    # Retrieval / context packing
    TOP_K_QA: int = 3
    TOP_K_JD: int = 3
    MAX_CHUNK_CHARS_IN_PROMPT: int = 600

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:4b"
    OLLAMA_TIMEOUT_S: int = 600

    # Debug/Log
    DEBUG: bool = False
    LOG_TIMING: bool = True
    LOG_LEVEL: str = "INFO" # "INFO" / "WARNING" / "DEBUG"
    QUIET_LIBS: bool = True


SETTINGS = Settings()