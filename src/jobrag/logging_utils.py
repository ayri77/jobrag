import logging

def configure_logging(level: str = "INFO", quiet_libs: bool = True) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if quiet_libs:
        for name in ["httpx", "huggingface_hub", "sentence_transformers", "transformers", "urllib3"]:
            logging.getLogger(name).setLevel(logging.WARNING)