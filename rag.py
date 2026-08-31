import os
import json
import time
import pandas as pd

from datasets import Dataset

import config
from test_data import get_test_dataset
from rag import query_rag

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ============================================================
# 1. PREPARE DATASET
# ============================================================

def prepare_evaluation_dataset():

    test_cases = get_test_dataset()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n" + "=" * 70)
    print("RUNNING RAG PIPELINE FOR EVALUATION DATASET")
    print("=" * 70)

    for i, sample in enumerate(test_cases, 1):

        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"\n[{i}/{len(test_cases)}]")
        print(f"Question: {question}")

        result = query_rag(question)

        answer = result.get("answer", "")
        retrieved = result.get("retrieved_contexts", [])

        if not isinstance(retrieved, list):
            retrieved = [str(retrieved)]

        retrieved = [
            str(context).strip()
            for context in retrieved
            if str(context).strip()
        ]

        print(f"Answer: {answer[:200]}...")
        print(f"Retrieved contexts: {len(retrieved)}")

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved)
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    return dataset


# ============================================================
# 2. CREATE LOCAL OLLAMA EVALUATOR
# ============================================================

def create_evaluator():

    print("\n" + "=" * 70)
    print("CREATING LOCAL OLLAMA EVALUATOR")
    print("=" * 70)

    print(f"LLM Model       : {config.LLM_MODEL}")
    print(f"Embedding Model : {config.EMBEDDING_MODEL}")
    print(f"Ollama URL      : {config.OLLAMA_BASE_URL}")

    # IMPORTANT:
    # Do NOT use format="json" here.
    # RAGAS itself controls the structured evaluation prompts.
    evaluator_chat = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=8192,
        num_predict=2048,
    )

    evaluator_llm = LangchainLLMWrapper(
        evaluator_chat
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    )

    return evaluator_llm, evaluator_embeddings


# ============================================================
# 3. RUN RAGAS
# ============================================================

def run_ragas_evaluation():

    print("\n" + "=" * 70)
    print("                 GFR RAGAS EVALUATION")
    print("=" * 70)

    dataset = prepare_evaluation_dataset()

    print(f"\nNumber of questions: {len(dataset)}")

    evaluator_llm, evaluator_embeddings = create_evaluator()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Give RAGAS enough time and retries because Ollama is local.
    run_config = RunConfig(
        timeout=180,
        max_retries=5,
        max_wait=30,
    )

    print("\n" + "=" * 70)
    print("RAGAS METRICS")
    print("=" * 70)

    print("1. Faithfulness")
    print("2. Answer Relevancy")
    print("3. Context Precision")
    print("4. Context Recall")

    print("\nStarting evaluation...")
    print("IMPORTANT: Local Ollama evaluation can take several minutes.")
    print("Do NOT close the terminal while evaluation is running.\n")

    start_time = time.time()

    try:

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,

            # IMPORTANT:
            # One metric failing should NOT destroy the complete evaluation.
            raise_exceptions=False,

            show_progress=True,
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("RAGAS EVALUATION FAILED")
        print("=" * 70)

        print(f"\nError type : {type(e).__name__}")
        print(f"Error     : {e}")

        print("\nThe RAG pipeline itself is not necessarily broken.")
        print("This error occurred inside the RAGAS evaluator.")

        return None

    elapsed = time.time() - start_time

    # ========================================================
    # 4. SAVE RESULTS
    # ========================================================

    df = results.to_pandas()

    os.makedirs(
        config.EVALUATION_RESULTS_DIR,
        exist_ok=True
    )

    csv_path = os.path.join(
        config.EVALUATION_RESULTS_DIR,
        "ragas_results.csv"
    )

    json_path = os.path.join(
        config.EVALUATION_RESULTS_DIR,
        "ragas_results.json"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            df.to_dict(orient="records"),
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # ========================================================
    # 5. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("                 RAGAS RESULTS")
    print("=" * 70)

    print(f"\nEvaluation time: {elapsed / 60:.2f} minutes\n")

    print(df.to_string(index=False))

    print("\n" + "-" * 70)
    print("AVERAGE METRIC SCORES")
    print("-" * 70)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for metric in metric_names:

        if metric in df.columns:

            score = pd.to_numeric(
                df[metric],
                errors="coerce"
            ).mean()

            if pd.isna(score):

                print(f"{metric:<20}: FAILED / NaN")

            else:

                print(
                    f"{metric:<20}: {score:.4f}"
                )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print("\nSaved files:")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")

    return df


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    run_ragas_evaluation()