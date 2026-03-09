from jobrag.embed import EmbeddingModel
from jobrag.store import FaissStore

def main():
    texts = [
        "I built ML pipelines and deployed models with FastAPI and Docker.",
        "I led enterprise BI and data platforms for logistics and agriculture.",
        "I have experience with Microsoft Graph API and workflow automation.",
        "I enjoy astronomy and physics in my free time.",
    ]

    embedder = EmbeddingModel()
    doc_vecs = embedder.embed(texts)

    store = FaissStore.new_cosine(dim=doc_vecs.shape[1])
    store.add(doc_vecs)

    query = "I developed backend APIs and deployed ML services."
    qv = embedder.embed([query])

    scores, idx = store.search(qv[0], top_k=3)

    print("Query:", query)
    print("\nTop matches:")
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        print(f"{rank}. score={s:.4f} | {texts[i]}")

if __name__ == "__main__":
    main()