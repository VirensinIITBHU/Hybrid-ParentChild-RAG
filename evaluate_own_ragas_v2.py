"""
evaluate_own_ragas_v2.py

Mini RAGAS-style Evaluation

Metrics:
- Retrieval Recall@K
- Answer Similarity
- Faithfulness
- Context Precision
"""

import json
import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from generate import (
    generate_answer_with_context
)

from retrieve_reranked_hybrid import (
    retrieve_hybrid
)

TOP_K = 10

DATASET_PATH = (
    "data/evaluation/qa_set.json"
)

eval_model = None


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def load_eval_model():

    global eval_model

    if eval_model is None:

        print(
            "Loading evaluation model..."
        )

        eval_model = (
            SentenceTransformer(
                "BAAI/bge-small-en-v1.5"
            )
        )

        print(
            "Evaluation model loaded."
        )

    return eval_model


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def similarity(
    text1,
    text2,
):

    model = load_eval_model()

    emb1 = model.encode(
        text1,
        normalize_embeddings=True,
    )

    emb2 = model.encode(
        text2,
        normalize_embeddings=True,
    )

    return float(
        cosine_similarity(
            [emb1],
            [emb2],
        )[0][0]
    )


def best_context_similarity(
    answer,
    contexts,
):

    if not contexts:
        return 0.0

    scores = [
        similarity(answer, ctx)
        for ctx in contexts
    ]

    return max(scores)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )["samples"]


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate():

    samples = load_dataset()

    print(
        f"Loaded {len(samples)} questions."
    )

    retrieval_hits = 0

    answer_similarity_scores = []

    faithfulness_scores = []

    context_precision_scores = []

    all_results = []

    for idx, sample in enumerate(
        samples,
        start=1,
    ):

        question = sample[
            "question"
        ]

        ground_truth = sample[
            "ground_truth"
        ]

        expected_parent = sample[
            "parent_id"
        ]

        print(
            f"\n[{idx}/{len(samples)}]"
        )

        print(
            f"Question: {question}"
        )

        # ------------------------------
        # Retrieval Recall
        # ------------------------------

        retrieved_docs = (
            retrieve_hybrid(
                query=question,
                k=TOP_K,
            )
        )

        retrieved_parents = [

            parent_id

            for (
                parent_id,
                _,
                _
            )

            in retrieved_docs
        ]

        hit = (
            expected_parent
            in retrieved_parents
        )

        if hit:

            retrieval_hits += 1

        # ------------------------------
        # Generation
        # ------------------------------

        try:

            result = (
                generate_answer_with_context(
                    question
                )
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

            continue

        answer = result["answer"]

        contexts = result["contexts"]

        docs = result["retrieved_docs"]

        # ------------------------------
        # Metrics
        # ------------------------------

        answer_sim = similarity(
            answer,
            ground_truth,
        )

        faithfulness = (
            best_context_similarity(
                answer,
                contexts,
            )
        )

        relevant_docs = sum(
            1
            for (
                _,
                _,
                score,
            )
            in docs
            if score > 1.0
        )

        precision = (
            relevant_docs
            / len(docs)
        )

        answer_similarity_scores.append(
            answer_sim
        )

        faithfulness_scores.append(
            faithfulness
        )

        context_precision_scores.append(
            precision
        )

        all_results.append(
            {
                "question":
                    question,

                "hit":
                    hit,

                "answer_similarity":
                    answer_sim,

                "faithfulness":
                    faithfulness,

                "context_precision":
                    precision,
            }
        )

        print(
            f"Recall Hit: {hit}"
        )

        print(
            f"Answer Similarity: "
            f"{answer_sim:.4f}"
        )

        print(
            f"Faithfulness: "
            f"{faithfulness:.4f}"
        )

        print(
            f"Context Precision: "
            f"{precision:.4f}"
        )

    return (
        samples,
        retrieval_hits,
        answer_similarity_scores,
        faithfulness_scores,
        context_precision_scores,
        all_results,
    )


# --------------------------------------------------
# REPORT
# --------------------------------------------------

def generate_report(
    samples,
    retrieval_hits,
    answer_similarity_scores,
    faithfulness_scores,
    context_precision_scores,
    all_results,
):

    retrieval_recall = (
        retrieval_hits
        / len(samples)
    )

    avg_answer_similarity = np.mean(
        answer_similarity_scores
    )

    avg_faithfulness = np.mean(
        faithfulness_scores
    )

    avg_context_precision = np.mean(
        context_precision_scores
    )

    print(
        "\n========================"
    )

    print(
        "MINI RAGAS V2"
    )

    print(
        "========================\n"
    )

    print(
        f"Retrieval Recall@{TOP_K}: "
        f"{retrieval_recall:.4f}"
    )

    print(
        f"Answer Similarity: "
        f"{avg_answer_similarity:.4f}"
    )

    print(
        f"Faithfulness: "
        f"{avg_faithfulness:.4f}"
    )

    print(
        f"Context Precision: "
        f"{avg_context_precision:.4f}"
    )

    df = pd.DataFrame(
        all_results
    )

    worst = df.sort_values(
        "answer_similarity"
    ).head(5)

    best = df.sort_values(
        "answer_similarity",
        ascending=False,
    ).head(5)

    print(
        "\n========================"
    )

    print(
        "WORST 5 QUESTIONS"
    )

    print(
        "========================"
    )

    print(
        worst[
            [
                "question",
                "answer_similarity",
                "faithfulness",
            ]
        ]
    )

    print(
        "\n========================"
    )

    print(
        "BEST 5 QUESTIONS"
    )

    print(
        "========================"
    )

    print(
        best[
            [
                "question",
                "answer_similarity",
                "faithfulness",
            ]
        ]
    )

    df.to_csv(
        "mini_ragas_v2_results.csv",
        index=False,
    )

    print(
        "\nSaved: mini_ragas_v2_results.csv"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    results = evaluate()

    generate_report(*results)


if __name__ == "__main__":
    main()