import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "GFR.pdf")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ---- Local Ollama models (no paid API key needed) ----
# Pull these first in a terminal:
#   ollama pull llama3.1
#   ollama pull nomic-embed-text
LLM_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

# ---- Chunking / retrieval settings ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

COLLECTION_NAME = "gfr_docs"