from pathlib import Path
from jobrag.rag import RAGEngine

def main():
    project_root = Path(__file__).resolve().parents[2]
    index_dir = project_root / "data" / "index"
    rag = RAGEngine.from_index_dir(index_dir=index_dir)

    jd = """
We are looking for a Python/AI Engineer to build LLM-based assistants and RAG systems.
Requirements:
- Python, FastAPI
- Vector databases / embeddings, retrieval
- Docker, deployment
- Experience with Azure or cloud is a plus
"""

    summary, docs = rag.summarize_for_jd(jd, top_k=3, stream=False)

    print(summary)
    print("\n--- Retrieved ---")
    for d in docs:
        print(f"{d['score']:.4f} | {d['doc_id']} p{d['page_num']}")

if __name__ == "__main__":
    main()