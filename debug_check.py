"""
Diagnostic script — run this to see WHY the RAG pipeline says "I don't know".

    python debug_check.py
"""

from langchain_community.document_loaders import PyPDFLoader
import config
from rag_pipeline import get_retriever

print("=== STEP 1: Checking raw PDF text extraction ===")
loader = PyPDFLoader(config.PDF_PATH)
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

sample = pages[0].page_content.strip()
print(f"\nFirst page character count: {len(sample)}")
print("First 500 characters of page 1:\n")
print(sample[:500] if sample else "(EMPTY — this page has no extractable text)")

total_chars = sum(len(p.page_content.strip()) for p in pages)
avg_chars = total_chars / len(pages) if pages else 0
print(f"\nAverage characters per page across whole PDF: {avg_chars:.0f}")
if avg_chars < 50:
    print(">>> WARNING: Very little text extracted. This PDF is likely scanned/image-based")
    print(">>> and needs OCR before it can be used for RAG.")

print("\n=== STEP 2: Checking what gets retrieved for 'What is GFR?' ===")
retriever = get_retriever(k=4)
docs = retriever.invoke("What is GFR?")
print(f"Retrieved {len(docs)} chunk(s).\n")
for i, d in enumerate(docs, 1):
    print(f"--- Chunk {i} (page {d.metadata.get('page', '?')}) ---")
    print(d.page_content[:400])
    print()