
"""
retrieve_bm25.py

Purpose
--------
Lexical retrieval using BM25.

Chunk-Level Retrieval

BM25 excels at:

- Exact keywords
- Acronyms
- Paper names
- Rare terms
"""

import re

from rank_bm25 import BM25Okapi

from ingest import load_chunks

# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

print("Loading chunks...")

child_docs, _ = load_chunks()

print(
    f"Loaded {len(child_docs)} chunks."
)

# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    return text.split()

# --------------------------------------------------
# BUILD BM25 INDEX
# --------------------------------------------------

print("Building BM25 index...")

corpus = [
    tokenize(doc.page_content)
    for doc in child_docs
]

bm25 = BM25Okapi(
    corpus
)

print("BM25 ready.")

# --------------------------------------------------
# RETRIEVE
# --------------------------------------------------

def bm25_retrieve(
    query: str,
    k: int = 5,
):
    """
    Chunk-Level BM25 Retrieval
    """

    tokenized_query = tokenize(
        query
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for idx in ranked_indices[:k]:

        doc = child_docs[idx]

        results.append(
            {
                "score": float(
                    scores[idx]
                ),
                "document": doc,
            }
        )

    return results

# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = "What is LoRA?"

    results = bm25_retrieve(
        query=query,
        k=5,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['document'].metadata['chunk_id']}"
        )

        print(
            f"Paper: "
            f"{result['document'].metadata['paper_name']}"
        )

        print(
            result["document"]
            .page_content[:300]
        )