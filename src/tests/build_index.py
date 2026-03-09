from pathlib import Path
from jobrag.index import build_index
from jobrag.settings import SETTINGS

def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / SETTINGS.RAW_DIR
    output_dir = project_root / SETTINGS.INDEX_DIR

    build_index(raw_dir=raw_dir, output_dir=output_dir)

if __name__ == "__main__":
    main()