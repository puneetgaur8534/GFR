import os
import sys

import requests

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval / generation
TOP_K = 4
TEMPERATURE = 0.0

# Paths (relative to project root)
DATA_DIR = "./data"
PDF_PATH = os.path.join(DATA_DIR, "gfr.pdf")
VECTOR_STORE_DIR = "./vector_store"
EVALUATION_RESULTS_DIR = "./evaluation_results"
COLLECTION_NAME = "gfr_2017_rules"


def check_ollama(require_models: bool = False) -> None:
    """Verify Ollama is reachable and required models are available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            "ERROR: Ollama is not reachable at "
            f"{OLLAMA_BASE_URL}. Start Ollama and pull required models.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)

    if not require_models:
        return

    models = {item.get("name", "") for item in response.json().get("models", [])}
    normalized = {name.split(":")[0] for name in models}
    normalized |= models

    missing = []
    for model in (LLM_MODEL, EMBEDDING_MODEL):
        base = model.split(":")[0]
        if model not in models and base not in normalized:
            missing.append(model)

    if missing:
        print("ERROR: Required Ollama models are missing:", file=sys.stderr)
        for model in missing:
            print(f"  - {model}", file=sys.stderr)
        print("\nPull them with:", file=sys.stderr)
        for model in missing:
            print(f"  ollama pull {model}", file=sys.stderr)
        sys.exit(1)
