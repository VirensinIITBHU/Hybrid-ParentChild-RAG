"""
evaluate_hybrid.py

Parent Recall@K Evaluation

Question
    ↓
Hybrid Retrieval
    ↓
Parent Retrieval
    ↓
Was the parent containing
the gold chunk retrieved?
"""

import json

from ingest import load_chunks
from retrieve_reranked_hybrid import (
    retrieve_hybrid,
)


def load_evaluation_set():

    with open(
        "data/evaluation/qa_set.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)["samples"]


def evaluate_recall_at_k(
    evaluation_set,
    child_docs,
    k=20,
):

    hits = 0
    failed_queries = []

    for sample in evaluation_set:

        question = sample["question"]

        expected_chunk_id = (
            sample["chunk_id"]
        )

        expected_parent_id = (
            child_docs[
                expected_chunk_id
            ]
            .metadata["parent_id"]
        )

        results = retrieve_hybrid(
            query=question,
            k=k,
        )

        retrieved_parent_ids = [
            parent_id
            for (
                parent_id,
                _,
                _,
            )
            in results
        ]

        found = (
            expected_parent_id
            in retrieved_parent_ids
        )

        if found:

            hits += 1

        else:

            failed_queries.append(
                {
                    "question":
                        question,

                    "expected_chunk":
                        expected_chunk_id,

                    "expected_parent":
                        expected_parent_id,

                    "retrieved_parents":
                        retrieved_parent_ids,
                }
            )

    recall = hits / len(
        evaluation_set
    )

    return (
        recall,
        hits,
        failed_queries,
    )


def print_results(
    recall,
    hits,
    total,
    k,
    failed_queries,
):

    print("\n===================")

    print(
        f"Parent Recall@{k} = "
        f"{recall:.2%}"
    )

    print(
        f"Hits: "
        f"{hits}/{total}"
    )

    print("===================")

    print(
        f"\nFailed Queries: "
        f"{len(failed_queries)}"
    )

    for failure in failed_queries:

        print(
            "\n--------------------------------"
        )

        print(
            f"Question : "
            f"{failure['question']}"
        )

        print(
            f"Expected Chunk : "
            f"{failure['expected_chunk']}"
        )

        print(
            f"Expected Parent : "
            f"{failure['expected_parent']}"
        )

        print(
            f"Retrieved Parents : "
            f"{failure['retrieved_parents']}"
        )


def main():

    K = 10

    evaluation_set = (
        load_evaluation_set()
    )

    child_docs, parent_docs = (
        load_chunks()
    )

    (
        recall,
        hits,
        failed_queries,
    ) = evaluate_recall_at_k(
        evaluation_set=evaluation_set,
        child_docs=child_docs,
        k=K,
    )

    print_results(
        recall=recall,
        hits=hits,
        total=len(
            evaluation_set
        ),
        k=K,
        failed_queries=failed_queries,
    )


if __name__ == "__main__":
    main()