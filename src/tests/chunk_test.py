from pathlib import Path
from jobrag.ingest import ingest_dir
from jobrag.chunk import chunk_pages

def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    pages = ingest_dir(raw_dir)
    chunks = chunk_pages(pages, chunk_size=1200, chunk_overlap=200)

    print("Pages:", len(pages))
    print("Chunks:", len(chunks))

    # Show a few chunks from first doc
    first_doc = chunks[0].doc_id
    sample = [c for c in chunks if c.doc_id == first_doc][:3]
    for c in sample:
        print("\n---")
        print(f"{c.doc_id} p{c.page_num} chunk#{c.chunk_id} len={len(c.text)}")
        print(c.text[:400])

if __name__ == "__main__":
    main()