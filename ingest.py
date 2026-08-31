"""
Run this ONCE (and again any time you change the PDF) to build the vector store.

    python ingest.py

--------------------------------------------------------------------------
CHUNKING STRATEGY: rule-boundary-aware (upgraded from plain fixed-size
character splitting)
--------------------------------------------------------------------------
The GFR document is structured around numbered "Rule N ..." units — each
one is a self-contained rule of law. Plain RecursiveCharacterTextSplitter
with a fixed chunk_size doesn't know this, so a 1000-character window
regularly ends up containing the tail of one rule plus the start of an
unrelated next rule. That noise is a major reason RAGAS's
context_precision score comes back low: the retriever sometimes returns
chunks that are only partially about the question's actual rule.

This version instead:
  1. Finds every "Rule <number>" heading in the extracted text using a
     regex, and treats the text between one "Rule N" and the next as one
     logical unit.
  2. Tags each resulting chunk with metadata (rule_number, page) so you
     can inspect/debug retrieval later, or add metadata filtering.
  3. If a single rule's text is unusually long (some GFR rules run several
     pages with sub-clauses), it's further split with
     RecursiveCharacterTextSplitter so no single chunk becomes too large
     for the embedding model / context window.
  4. Any text before "Rule 1" (title page, table of contents, preamble)
     is chunked with the original RecursiveCharacterTextSplitter as a
     fallback, since it has no rule structure to align to.

This keeps each chunk aligned with one real rule's boundaries, which
should directly improve context_precision without needing any paid API
or external tool — it's pure regex + your existing local embeddings.
"""

import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

# Matches "Rule 130", "Rule25", "Rule 25   (1)" etc. at the start of a
# rule heading. GFR text extraction sometimes drops/adds spacing
# inconsistently (see Rule20, Rule25 vs Rule 130 in the raw PDF text),
# so this is deliberately lenient about whitespace between "Rule" and
# the number.
RULE_HEADING_PATTERN = re.compile(r"(Rule\s*\d+[A-Za-z]?)\b")

# If a single rule's raw text exceeds this many characters, it gets
# sub-split further so no chunk is excessively large for the embedding
# model or the LLM's context window.
MAX_RULE_CHUNK_SIZE = 1800
RULE_SUBSPLIT_OVERLAP = 150

# GFR's Rule 2 ("Definition: In these rules, unless the context otherwise
# requires-") is a single block containing ~40 separate term definitions,
# formatted as "(i) "Accounts Officer" means...", "(ii) "Administrator"
# means...", etc. If this whole block is sub-split by raw character count
# like any other long rule, each resulting chunk ends up containing 8-10
# UNRELATED definitions mashed together. Embedding a chunk like that
# produces a blurry "average" vector that doesn't strongly represent any
# single term - which is exactly why a query like "Competent Authority
# means" was matching unrelated rules (301, 180, 132 - which happen to
# use the phrase "competent authority" a lot) instead of the actual
# definition in Rule 2. This pattern detects that per-item structure and
# splits Rule 2 into one small, precise chunk per definition instead.
DEFINITION_ITEM_PATTERN = re.compile(
    r"\((?:[ivxlcdm]+)\)\s*[“\"]"  # e.g. "(v) “" or "(v) \""
)
# Extracts the defined term itself, e.g. "Competent Authority", so it can
# be stored in metadata for easier debugging/inspection later.
DEFINED_TERM_PATTERN = re.compile(r"[“\"]([^”\"]+)[”\"]\s*means")


def _extract_full_text_with_pages(documents) -> list[tuple[str, int]]:
    """Returns a list of (page_text, page_number) tuples from the loaded PDF pages."""
    return [(doc.page_content, doc.metadata.get("page", -1)) for doc in documents]


def _split_definitions_block(rule_text: str, rule_label: str, source_page: int) -> list[Document]:
    """
    Splits a definitions-style rule (like GFR's Rule 2) into one chunk
    per individual "(i) "Term" means..." item, instead of blind
    character-count splitting. This keeps each definition's embedding
    sharply focused on that one term, instead of being diluted by
    several unrelated definitions sharing one chunk.
    """
    item_matches = list(DEFINITION_ITEM_PATTERN.finditer(rule_text))
    docs: list[Document] = []

    # Keep the block's own intro line (e.g. "Rule 2 Definition: In these
    # rules, unless the context otherwise requires-") attached to the
    # FIRST item only, so that context isn't lost, but every other item
    # stands alone.
    for idx, match in enumerate(item_matches):
        start = match.start()
        end = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(rule_text)
        item_text = rule_text[start:end].strip()

        if idx == 0:
            intro = rule_text[:start].strip()
            if intro:
                item_text = f"{intro}\n{item_text}"

        if not item_text:
            continue

        term_match = DEFINED_TERM_PATTERN.search(item_text)
        defined_term = term_match.group(1) if term_match else None

        docs.append(
            Document(
                page_content=item_text,
                metadata={
                    "rule_number": rule_label,
                    "page": source_page,
                    "defined_term": defined_term,
                },
            )
        )

    return docs


def _build_rule_aware_chunks(pages: list[tuple[str, int]]) -> list[Document]:
    """
    Joins all page text together (tracking approximate page boundaries),
    finds every "Rule N" heading, and creates one Document per rule
    (sub-splitting only if that rule's text is unusually long).
    """
    # Join pages with a page-boundary marker so we can still recover an
    # approximate source page per rule afterward.
    full_text_parts = []
    page_offsets = []  # (char_offset_in_full_text, page_number)
    running_offset = 0
    for page_text, page_number in pages:
        page_offsets.append((running_offset, page_number))
        full_text_parts.append(page_text)
        running_offset += len(page_text) + 1  # +1 for the join separator below
    full_text = "\n".join(full_text_parts)

    matches = list(RULE_HEADING_PATTERN.finditer(full_text))

    chunks: list[Document] = []

    def _page_for_offset(offset: int) -> int:
        page_number = pages[0][1] if pages else -1
        for char_offset, pnum in page_offsets:
            if char_offset <= offset:
                page_number = pnum
            else:
                break
        return page_number

    # Anything before the first "Rule N" heading (title page, TOC,
    # preamble, definitions intro) has no rule structure to align to —
    # fall back to plain recursive character splitting for that part.
    if matches:
        preamble_text = full_text[: matches[0].start()].strip()
    else:
        preamble_text = full_text.strip()

    if preamble_text:
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        for piece in fallback_splitter.split_text(preamble_text):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={"rule_number": "preamble", "page": pages[0][1] if pages else -1},
                )
            )

    # Build one chunk per rule: text from this "Rule N" heading up to
    # (not including) the next "Rule N" heading.
    for i, match in enumerate(matches):
        rule_label = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        rule_text = full_text[start:end].strip()

        if not rule_text:
            continue

        source_page = _page_for_offset(start)

        # Check if this rule looks like a definitions block (GFR's
        # Rule 2 specifically) - i.e. it contains several "(i) "Term"
        # means..." items. If so, split by individual definition
        # instead of by raw character count, regardless of the rule's
        # total length.
        definition_item_count = len(DEFINITION_ITEM_PATTERN.findall(rule_text))

        if definition_item_count >= 3:
            chunks.extend(_split_definitions_block(rule_text, rule_label, source_page))
        elif len(rule_text) <= MAX_RULE_CHUNK_SIZE:
            chunks.append(
                Document(
                    page_content=rule_text,
                    metadata={"rule_number": rule_label, "page": source_page},
                )
            )
        else:
            # Long rule (many sub-clauses) — split further, but keep the
            # rule_number tag on every sub-chunk so retrieval/debugging
            # still knows which rule they came from.
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_RULE_CHUNK_SIZE,
                chunk_overlap=RULE_SUBSPLIT_OVERLAP,
            )
            for j, piece in enumerate(sub_splitter.split_text(rule_text)):
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "rule_number": rule_label,
                            "page": source_page,
                            "part": j + 1,
                        },
                    )
                )

    return chunks


def build_vectorstore():
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at: {config.PDF_PATH}\n"
            f"Put your GFR PDF inside the 'data' folder and name it GFR.pdf"
        )

    print("Loading PDF...")
    loader = PyPDFLoader(config.PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s).")

    print("Splitting into rule-aware chunks...")
    pages = _extract_full_text_with_pages(documents)
    chunks = _build_rule_aware_chunks(pages)
    print(f"Created {len(chunks)} chunk(s) aligned to GFR rule boundaries.")

    # Wipe any old vector store so re-running this script never mixes old + new data
    if os.path.exists(config.VECTORSTORE_DIR):
        shutil.rmtree(config.VECTORSTORE_DIR)

    print("Embedding chunks locally with Ollama (this can take a few minutes)...")
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTORSTORE_DIR,
    )

    print(f"Done. Vector store saved at: {config.VECTORSTORE_DIR}")
    return vectordb


if __name__ == "__main__":
    build_vectorstore()