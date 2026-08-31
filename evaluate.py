"""
Evaluates the Stack Board's RAG pipeline with RAGAS, using local Ollama
models as the judge (no paid API key needed).

Metrics: faithfulness, answer relevancy, context precision, semantic (answer) similarity.

Run with:
    python evaluate.py
"""

# --- Compatibility shim -----------------------------------------------
# ragas tries to import langchain_community.chat_models.vertexai at
# startup even though we never use Google VertexAI. Newer versions of
# langchain-community removed that module, which crashes the import
# with: ModuleNotFoundError: No module named
# 'langchain_community.chat_models.vertexai'
# Since we only use local Ollama models, we register a harmless
# placeholder module so ragas's import succeeds. This must run BEFORE
# any ragas import.
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

from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_similarity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config
from rag_pipeline import answer_query
from ground_truth import ground_truth_data


# ---------------------------------------------------------------------
# Evaluator model settings
# ---------------------------------------------------------------------
# IMPORTANT: this does NOT have to be the same model your chatbot uses
# to answer questions (config.LLM_MODEL). Faithfulness and
# context_precision make the LLM do multi-step structured reasoning
# (break the answer into statements, verify each one, return JSON-like
# verdicts) — several LLM calls per single question. On local CPU-only
# Ollama, a bigger model here can be so slow that RAGAS's per-call
# timeout gets hit before it ever finishes, which is the TimeoutError
# you saw. A smaller, faster model is usually MORE reliable as an
# evaluator, not less, for this reason.
#
# Pull this once in a terminal if you haven't already:
#   ollama pull qwen2.5:7b-instruct
#
# Why qwen2.5 instead of llama3.2:3b or llama3.1:
# - llama3.2:3b was fast but too weak to reliably follow RAGAS's strict
#   JSON-output instructions — it echoed the prompt's own schema back as
#   its "answer" instead of generating real JSON (see the
#   OutputParserException you hit).
# - llama3.1:8b could plausibly follow the format but was too slow on
#   CPU-only local inference, hitting the timeout before finishing.
# - qwen2.5:7b-instruct is specifically tuned for strong instruction-
#   following and structured/JSON output, which is exactly what RAGAS's
#   faithfulness and context_precision prompts require. It's a good
#   middle ground: capable enough to follow the format, still small
#   enough to run reasonably on local hardware.
EVALUATOR_LLM_MODEL = "qwen2.5:7b-instruct"


def build_eval_dataset() -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in ground_truth_data:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"Running RAG pipeline for: {q}")
        result = answer_query(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(gt)

    # IMPORTANT: newer versions of ragas expect these exact column names
    # (user_input / retrieved_contexts / response / reference), NOT the
    # older question / contexts / answer / ground_truth naming. Using the
    # old names doesn't raise an error - ragas silently reinterprets the
    # columns internally, which is what caused context_precision to
    # score 0.0 across the board despite retrieval actually being good.
    # The isolated test script (test_context_precision.py) used the
    # correct new names directly and scored ~1.0 on the same kind of
    # data, confirming this was the mismatch.
    data = {
        "user_input": questions,
        "retrieved_contexts": contexts,
        "response": answers,
        "reference": ground_truths,
    }
    return Dataset.from_dict(data)


def run_diagnostic_single_context_precision(dataset: Dataset, evaluator_llm, evaluator_embeddings, run_config):
    """
    Runs context_precision separately for EACH individual retrieved
    context (instead of all 4 at once) on every question. This tells us
    whether ANY single retrieved chunk is being judged useful, and
    which one - useful for confirming whether the Rule 2 definitions
    chunking fix actually improved retrieval, independent of any
    quirks in how the metric handles multi-context lists at once.
    """
    print("\n=== DIAGNOSTIC: per-context precision (one context at a time) ===")
    for i in range(len(dataset)):
        row = dataset[i]
        print(f"\n--- Question {i}: {row['user_input']} ---")
        for j, ctx in enumerate(row["retrieved_contexts"]):
            single_ctx_data = {
                "user_input": [row["user_input"]],
                "retrieved_contexts": [[ctx]],
                "response": [row["response"]],
                "reference": [row["reference"]],
            }
            single_ds = Dataset.from_dict(single_ctx_data)
            try:
                single_result = evaluate(
                    dataset=single_ds,
                    metrics=[context_precision],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                )
                score = single_result.to_pandas()["context_precision"].iloc[0]
            except Exception as e:
                score = f"ERROR: {e}"
            print(f"  context[{j}] precision={score} | rule={ctx[:60]!r}...")
    print("=== END DIAGNOSTIC ===\n")


def run_evaluation():
    dataset = build_eval_dataset()

    # DEBUG: print the actual dataset content so we can compare it
    # against the isolated test_context_precision.py example that
    # scored ~1.0. This will show if something in the real
    # question/answer/context/reference content is structurally
    # different from the hand-crafted test case.
    print("\n=== DEBUG: dataset being sent to evaluate() ===")
    for i in range(len(dataset)):
        row = dataset[i]
        print(f"\n--- Row {i} ---")
        print("user_input:", repr(row["user_input"]))
        print("response:", repr(row["response"])[:300])
        print("reference:", repr(row["reference"])[:300])
        print("retrieved_contexts count:", len(row["retrieved_contexts"]))
        for j, ctx in enumerate(row["retrieved_contexts"]):
            print(f"  context[{j}]:", repr(ctx)[:200])
    print("=== END DEBUG ===\n")

    # Local judge model + local embeddings — nothing paid, nothing external
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=EVALUATOR_LLM_MODEL,
            temperature=0,
            num_predict=512,
            num_ctx=4096,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=config.EMBED_MODEL))

    metrics = [faithfulness, answer_relevancy, context_precision, answer_similarity]

    # RAGAS kills any single metric call that runs past `timeout` seconds
    # and records it as a failure (NaN, or a raised TimeoutError when
    # raise_exceptions=True). Local CPU-bound Ollama models are much
    # slower than a hosted API. max_workers=1 avoids firing multiple
    # requests at Ollama simultaneously, which would otherwise slow every
    # request down further and make timeouts more likely, not less.
    run_config = RunConfig(
        timeout=1200,      # 20 minutes per metric call — qwen2.5:7b is bigger than llama3.2:3b, give it room
        max_workers=1,     # one request to Ollama at a time
    )

    print(f"\nUsing '{EVALUATOR_LLM_MODEL}' as the evaluator/judge model.")
    print("Running RAGAS evaluation (this can take a while on local models)...\n")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        # TEMPORARY: surfaces the real exception behind any NaN instead of
        # silently swallowing it. Remove this once faithfulness/
        # context_precision reliably return real numbers, if you'd rather
        # RAGAS silently skip occasional failures as NaN instead.
        raise_exceptions=True,
    )

    df = results.to_pandas()
    print(df)

    output_path = "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path}")

    print("\n=== Average Scores ===")
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        print(f"{col}: {df[col].mean():.3f}")


if __name__ == "__main__":
    run_evaluation()