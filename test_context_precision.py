"""
Isolated test: does context_precision give a real (non-zero) score on an
obviously-correct, hand-crafted example? If this also returns 0.0, the
metric computation itself is broken for this setup - not your retrieval,
not your chunking, not your data.

Run with:
    python test_context_precision.py
"""

# --- Compatibility shim -----------------------------------------------
# Same fix as in evaluate.py: ragas tries to import
# langchain_community.chat_models.vertexai at startup even though we
# never use Google VertexAI. Newer langchain-community versions removed
# that module, so we register a harmless placeholder before any ragas
# import happens.
import sys
import types

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _fake_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder only, never instantiated
        pass

    _fake_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_module
# ------------------------------------------------------------------------

from ragas import evaluate
from ragas.metrics import context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset

import config

data = {
    "user_input": [
        "What does Rule 26 say about the responsibility of Controlling Officer?"
    ],
    "retrieved_contexts": [
        [
            "Rule 26 Responsibility of Controlling Officer in respect of Budget "
            "allocation. The duties and responsibilities of a controlling officer "
            "in respect of funds placed at his disposal are to ensure expenditure "
            "does not exceed budget allocation."
        ]
    ],
    "response": [
        "Rule 26 requires the Controlling Officer to ensure expenditure does not "
        "exceed budget allocation."
    ],
    "reference": [
        "Rule 26 requires the Controlling Officer to ensure expenditure does not "
        "exceed the budget allocation and is incurred for the intended purpose."
    ],
}

ds = Dataset.from_dict(data)

llm = LangchainLLMWrapper(
    ChatOllama(
        model="qwen2.5:7b-instruct",
        temperature=0,
        num_predict=512,
        num_ctx=4096,
    )
)
emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=config.EMBED_MODEL))

print("Running isolated context_precision test...\n")
result = evaluate(
    dataset=ds,
    metrics=[context_precision],
    llm=llm,
    embeddings=emb,
    raise_exceptions=True,
)

df = result.to_pandas()
print(df.to_string())
print("\ncontext_precision score:", df["context_precision"].iloc[0])