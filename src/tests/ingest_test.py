from pathlib import Path
from jobrag.ingest import ingest_dir

def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    pages = ingest_dir(raw_dir)

    print("PDF files:", len(list(raw_dir.glob("*.pdf"))))
    print("Extracted pages:", len(pages))

    # show first 2 pages
    for p in pages[:2]:
        print("\n---")
        print(p.doc_id, "page", p.page_num)
        print(p.text[:500])


if __name__ == "__main__":
    main()