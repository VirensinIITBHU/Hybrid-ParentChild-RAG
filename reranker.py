from sentence_transformers import CrossEncoder

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.")


def rerank(
    query,
    candidates,
):

    pairs = []

    for _, parent_doc in candidates:

        pairs.append(
            (
                query,
                parent_doc.page_content,
            )
        )

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for (
        parent_id,
        parent_doc,
    ), score in zip(
        candidates,
        scores,
    ):

        reranked.append(
            (
                parent_id,
                parent_doc,
                float(score),
            )
        )

    reranked.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return reranked