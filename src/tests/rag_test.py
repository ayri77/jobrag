from pathlib import Path
from jobrag.rag import RAGEngine

from jobrag.logging_utils import configure_logging
from jobrag.settings import SETTINGS

def main():
    project_root = Path(__file__).resolve().parents[2]
    index_dir = project_root / "data" / "index"

    rag = RAGEngine.from_index_dir(index_dir=index_dir)

    query = "What is my experience with FastAPI and Docker? Answer in English."
    answer, docs = rag.answer(query, top_k=3, stream=True)
    print(answer)
    print("\n--- Retrieved ---")
    for d in docs:
        print(f"{d['score']:.4f} | {d['doc_id']} p{d['page_num']}")

if __name__ == "__main__":
    configure_logging(SETTINGS.LOG_LEVEL, SETTINGS.QUIET_LIBS)
    main()