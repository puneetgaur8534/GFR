import os
import re
import shutil
import sys
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from config import check_ollama


def extract_rule_number(text: str) -> str:
    match = re.search(r"Rule\s+(\d+)", text, re.IGNORECASE)
    if match:
        return f"Rule {match.group(1)}"
    return "General"


def load_documents() -> List[Document]:
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at '{config.PDF_PATH}'. Place gfr.pdf inside data/."
        )

    print("Loading PDF...")
    loader = PyPDFLoader(config.PDF_PATH)
    return loader.load()


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\nRule ", "\nRule ", "\nCHAPTER ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["rule_number"] = extract_rule_number(chunk.page_content)
        chunk.metadata["chunk_id"] = f"chunk_{idx}"
        chunk.metadata["source"] = "GFR 2017"

    print(f"Documents/chunks created: {len(chunks)}")
    return chunks


def reset_vector_store() -> None:
    if os.path.isdir(config.VECTOR_STORE_DIR):
        shutil.rmtree(config.VECTOR_STORE_DIR)


def ingest_to_chroma(chunks: List[Document]) -> None:
    print("Connecting to Ollama...")
    check_ollama(require_models=True)

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    print("Creating embeddings...")
    print("Storing vectors...")
    reset_vector_store()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.VECTOR_STORE_DIR,
        collection_name=config.COLLECTION_NAME,
    )


def run_ingestion() -> None:
    try:
        documents = load_documents()
        chunks = chunk_documents(documents)
        ingest_to_chroma(chunks)
        print("Ingestion completed successfully.")
    except Exception as exc:
        print(f"ERROR: Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_ingestion()
