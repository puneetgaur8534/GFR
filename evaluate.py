import os
import json
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

from langchain_ollama import ChatOllama, OllamaEmbeddings


def prepare_evaluation_dataset():
    test_cases = get_test_dataset()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\nRunning RAG on evaluation dataset...\n")

    for i, sample in enumerate(test_cases, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"[{i}/{len(test_cases)}] {question}")

        result = query_rag(question)

        questions.append(question)
        answers.append(result["answer"])

        # IMPORTANT:
        # RAGAS expects a list of text contexts.
        retrieved = result.get("retrieved_contexts", [])

        if isinstance(retrieved, str):
            retrieved = [retrieved]

        contexts.append(retrieved)
        ground_truths.append(ground_truth)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_ragas_evaluation():

    print("\n" + "=" * 70)
    print("              GFR RAGAS EVALUATION")
    print("=" * 70)

    dataset = prepare_evaluation_dataset()

    print("\nCreating local Ollama evaluator...")

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("\nRunning RAGAS metrics...")
    print("This can take some time because Ollama is evaluating locally.\n")

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = results.to_pandas()

    os.makedirs(config.EVALUATION_RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(
        config.EVALUATION_RESULTS_DIR,
        "ragas_results.csv",
    )

    json_path = os.path.join(
        config.EVALUATION_RESULTS_DIR,
        "ragas_results.json",
    )

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            df.to_dict(orient="records"),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("                 RAGAS RESULTS")
    print("=" * 70)

    print(df.to_string(index=False))

    print("\nSaved:")
    print(csv_path)
    print(json_path)

    return df


if __name__ == "__main__":
    run_ragas_evaluation()