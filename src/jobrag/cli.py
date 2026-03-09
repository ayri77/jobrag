from __future__ import annotations

from pathlib import Path
import time
import typer
from rich import print as rprint

from jobrag.rag import RAGEngine
from jobrag.index import build_index
from jobrag.settings import SETTINGS
from jobrag.logging_utils import configure_logging

configure_logging(SETTINGS.LOG_LEVEL, SETTINGS.QUIET_LIBS)

app = typer.Typer(add_completion=False)


def _project_root() -> Path:
    # src/jobrag/cli.py -> project root
    return Path(__file__).resolve().parents[2]


def _index_dir() -> Path:
    return _project_root() / SETTINGS.INDEX_DIR


@app.command()
def ask(
    query: str = typer.Option(..., "--query", "-q"),
    top_k: int = typer.Option(SETTINGS.TOP_K_QA, "--top-k"),
    stream: bool = typer.Option(False, "--stream"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Ask a question over indexed documents (RAG)."""
    rag = RAGEngine.from_index_dir(index_dir=_index_dir())

    t0 = time.perf_counter()
    answer, docs = rag.answer(query, top_k=top_k, stream=stream)
    t1 = time.perf_counter()

    rprint(answer)
    rprint(f"\n[dim]Time: {t1 - t0:.2f}s[/dim]")

    if debug:
        rprint("\n--- Retrieved ---")
        for d in docs:
            rprint(f"{d['score']:.4f} | {d['doc_id']} p{d['page_num']}")


@app.command()
def jd(
    jd_text: str = typer.Option(None, "--jd", help="Job description text."),
    jd_file: Path = typer.Option(None, "--jd-file", exists=True, readable=True, help="Path to a JD text file."),
    top_k: int = typer.Option(SETTINGS.TOP_K_JD, "--top-k"),
    stream: bool = typer.Option(False, "--stream"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Generate a tailored candidate summary for a job description."""
    if (jd_text is None) == (jd_file is None):
        raise typer.BadParameter("Provide exactly one of --jd or --jd-file.")

    jd = jd_text if jd_text is not None else jd_file.read_text(encoding="utf-8")

    rag = RAGEngine.from_index_dir(index_dir=_index_dir())

    t0 = time.perf_counter()
    summary, docs = rag.summarize_for_jd(jd, top_k=top_k, stream=stream)
    t1 = time.perf_counter()

    rprint(summary)
    rprint(f"\n[dim]Time: {t1 - t0:.2f}s[/dim]")

    if debug:
        rprint("\n--- Retrieved ---")
        for d in docs:
            rprint(f"{d['score']:.4f} | {d['doc_id']} p{d['page_num']}")


@app.command()
def index(
    input_dir: Path = typer.Option(SETTINGS.RAW_DIR, "--input"),
    output_dir: Path = typer.Option(SETTINGS.INDEX_DIR, "--out"),
):
    """Build FAISS index from raw documents."""
    build_index(raw_dir=input_dir, output_dir=output_dir)

def main() -> None:
    app()


if __name__ == "__main__":
    main()