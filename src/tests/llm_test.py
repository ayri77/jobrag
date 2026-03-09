from jobrag.llm import OllamaClient
import time


def main():
    llm = OllamaClient(model="qwen3.5")
    start = time.perf_counter()
    data = llm.generate("Explain RAG in one sentence. Do not show reasoning.")
    wall = time.perf_counter() - start

    print(data["response"].strip())
    print(f"\nWall time: {wall:.2f}s")
    for k in ["total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]:
        if k in data:
            print(f"{k}: {data[k]}")

if __name__ == "__main__":
    main()