from jobrag.embed import EmbeddingModel

def main():
    model = EmbeddingModel()

    texts = [
        "I have experience with machine learning and LLM fine-tuning.",
        "My background includes Python backend development.",
    ]

    embeddings = model.embed(texts)

    print("Shape:", embeddings.shape)
    print("First vector (first 5 dims):", embeddings[0][:5])

if __name__ == "__main__":
    main()