from __future__ import annotations
import logging
import os
import re
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Supported text file extensions for chunking
SUPPORTED_TEXT_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".mdown", ".mkd", ".mkdn",
    ".rst", ".org",
    ".tex", ".latex", ".ltx", ".bib",
    ".json", ".jsonl", ".ndjson",
    ".csv", ".tsv",
    ".xml", ".xsl", ".xslt", ".xsd", ".dtd",
    ".html", ".htm", ".xhtml", ".shtml", ".stm", ".htmx",
    ".css", ".scss", ".sass", ".less", ".styl", ".stylus",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx", ".mts", ".cts",
    ".java",
    ".cpp", ".cc", ".cxx", ".c++", ".cp", ".C", ".h", ".hh", ".hpp", ".hxx", ".h++",
    ".c", ".h",
    ".cs",
    ".go",
    ".rs",
    ".rb", ".erb", ".erubis",
    ".py", ".pyw",
    ".sh", ".bash", ".zsh", ".fish", ".ksh", ".tcsh", ".csh",
    ".pl", ".pm", ".pod", ".t", ".plx", ".perl",
    ".php", ".php3", ".php4", ".php5", ".php7", ".phps", ".phpt",
    ".yaml", ".yml",
    ".toml",
    ".ini", ".cfg", ".conf", ".properties",
    ".env",
    ".gitignore", ".gitattributes", ".gitkeep",
    ".editorconfig",
    ".dockerfile", ".dockerignore",
)

def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 400,
    overlap: int = 80
) -> list[dict]:
    """Returns a list of dicts:
        {"source": str, "content": str, "chunk_idx": int}
        where each dict represents a chunk of the original text, along with its source and index.
        to access chunks -> chunks = chunks[i]["content"]
        """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        strip_whitespace=True,
    )

    chunks = splitter.split_text(text)
    logger.debug("Chunked '%s' into %d chunks", source, len(chunks))

    return [
        {"source": source, "content": chunk, "chunk_idx": idx}
        for idx, chunk in enumerate(chunks)
    ]

#=============UTILS=============
def get_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

import json
import os

TRACKING_FILE = "data/processed_files.json"


def load_tracking():
    if not os.path.exists(TRACKING_FILE):
        return {}
    with open(TRACKING_FILE, "r") as f:
        return json.load(f)


def save_tracking(data):
    os.makedirs("data", exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


#===============================

def load_corpus_from_directory(directory: str) -> list[dict]:
    corpus: list[dict] = []

    if not os.path.isdir(directory):
        return corpus

    tracking = load_tracking()
    updated_tracking = {}

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(SUPPORTED_TEXT_EXTENSIONS):
            continue

        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            file_hash = get_file_hash(text)

            if filename in tracking and tracking[filename] == file_hash:
                updated_tracking[filename] = file_hash
                continue

            chunks = chunk_text(text, source=filename)
            corpus.extend(chunks)

            updated_tracking[filename] = file_hash

        except OSError as exc:
            logger.warning("Could not read '%s': %s", filepath, exc)

    # Save updated tracking
    save_tracking(updated_tracking)
    logger.info("Loaded corpus: %d chunks from %s", len(corpus), directory)

    return corpus