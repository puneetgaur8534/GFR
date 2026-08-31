"""
Diagnostic: find where 'Competent Authority' is actually DEFINED in the
raw GFR PDF text (not just mentioned), so we can check whether the
Definitions section is being captured correctly by ingest.py's chunking.

Run with:
    python debug_definitions.py
"""

from langchain_community.document_loaders import PyPDFLoader
import config

loader = PyPDFLoader(config.PDF_PATH)
documents = loader.load()
full_text = "\n".join([d.page_content for d in documents])

search_term = "Competent Authority"
idx = full_text.find(search_term)

print(f"First mention of '{search_term}' at char index: {idx}")
print()
print(full_text[max(0, idx - 500): idx + 500])
print()
print("=" * 70)

# Also check for a "Definitions" heading anywhere in the document
def_idx = full_text.find("Definitions")
print(f"\nFirst mention of 'Definitions' at char index: {def_idx}")
if def_idx != -1:
    print(full_text[max(0, def_idx - 200): def_idx + 800])

print()
print("=" * 70)
print("\nText immediately surrounding the definitions block (char 800-1200):")
print(repr(full_text[800:1200]))

print()
print("=" * 70)
print("\nVery start of document (char 0-800), to find how this section begins:")
print(repr(full_text[0:800]))