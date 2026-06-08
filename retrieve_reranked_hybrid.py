from collections import defaultdict

from ingest import load_chunks
from reranker import rerank
from retrieve import retrieve
from retrieve_bm25 import bm25_retrieve


child_docs, parent_docs = load_chunks()


# --------------------------------------------------
# RRF
# --------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    rrf_k=60,
):

    fused_scores = defaultdict(float)

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = (
            result["document"]
            .metadata["chunk_id"]
        )

        fused_scores[
            chunk_id
        ] += 1 / (
            rrf_k + rank
        )

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        chunk_id = (
            result.payload[
                "chunk_id"
            ]
        )

        fused_scores[
            chunk_id
        ] += 1 / (
            rrf_k + rank
        )

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


# --------------------------------------------------
# HYBRID RETRIEVE
# --------------------------------------------------

def retrieve_hybrid(
    query: str,
    k: int = 10,
):

    bm25_results = bm25_retrieve(
        query=query,
        k=20,
    )

    dense_results = retrieve(
        query=query,
        k=20,
    )

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
    )

    # ------------------------------------------
    # CHILD -> PARENT
    # ------------------------------------------

    parent_candidates = {}

    for child_id, score in fused_results[:20]:

        parent_id = (
            child_docs[
                child_id
            ].metadata[
                "parent_id"
            ]
        )

        if parent_id not in parent_candidates:

            parent_candidates[parent_id] = score

        else:

            parent_candidates[parent_id] += score

    # ------------------------------------------
    # BUILD PARENT TEXTS
    # ------------------------------------------

    rerank_inputs = []

    for parent_id in parent_candidates:

        parent_doc = parent_docs[
            parent_id
        ]

        rerank_inputs.append(
            (
                parent_id,
                parent_doc,
            )
        )

    # ------------------------------------------
    # RERANK PARENTS
    # ------------------------------------------

    reranked = rerank(
        query=query,
        candidates=rerank_inputs,
    )

    return reranked[:k]


# --------------------------------------------------
# TEST queryy Below 
# --------------------------------------------------


if __name__ == "__main__":

    query = "What is LoRA?"

    results = retrieve_hybrid(
        query=query,
        k=5,
    )

    print("\nTop Parent Results\n")

    for rank, (
        parent_id,
        parent_doc,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        paper = parent_doc.metadata.get(
            "paper_name",
            "Unknown"
        )

        page = parent_doc.metadata.get(
            "page",
            "?"
        )

        print(
            f"{rank}. {paper}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Parent ID: {parent_id}"
        )

        print(
            parent_doc.page_content[:400]
        )

        print(
            f"\nReranker Score: "
            f"{score:.4f}"
        )

        print()