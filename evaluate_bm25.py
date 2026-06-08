
"""
evaluate_bm25.py

Measures BM25 Recall@K.

Chunk-Level Evaluation

Question
    ↓
BM25 Retrieval
    ↓
Top-K Chunks
    ↓
Was the correct chunk retrieved?
"""

import json

from retrieve_bm25 import bm25_retrieve

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

with open(
    "data/evaluation/qa_set.json",
    "r",
    encoding="utf-8",
) as f:

    evaluation_set = json.load(
        f
    )["samples"]

# --------------------------------------------------
# RECALL@K
# --------------------------------------------------

K = 10

hits = 0

failed_queries = []

for sample in evaluation_set:

    question = sample[
        "question"
    ]

    expected_chunk_id = sample[
        "chunk_id"
    ]

    results = bm25_retrieve(
        query=question,
        k=K,
    )

    retrieved_chunk_ids = []

    for result in results:

        chunk_id = (
            result["document"]
            .metadata["chunk_id"]
        )

        retrieved_chunk_ids.append(
            chunk_id
        )

    found = (
        expected_chunk_id
        in
        retrieved_chunk_ids
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

                "retrieved_chunks":
                    retrieved_chunk_ids,
            }
        )

# --------------------------------------------------
# FINAL SCORE
# --------------------------------------------------

recall = hits / len(
    evaluation_set
)

print("\n===================")

print(
    f"Recall@{K} = "
    f"{recall:.2%}"
)

print(
    f"Hits: "
    f"{hits}/"
    f"{len(evaluation_set)}"
)

print("===================")

# --------------------------------------------------
# FAILED QUERIES
# --------------------------------------------------

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
        f"Retrieved Chunks : "
        f"{failure['retrieved_chunks']}"
    )
