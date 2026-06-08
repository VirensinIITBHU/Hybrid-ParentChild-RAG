
"""
retrieve.py

Purpose
--------
Dense Retrieval using vector search.

Pipeline:

User Query(Hardcoded here to test)
    ↓
Query Embedding
    ↓
Qdrant Similarity Search
    ↓
Top-K Chunks
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from FlagEmbedding import FlagAutoModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

COLLECTION_NAME = "research_papers"

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-small-en-v1.5",
    query_instruction_for_retrieval=
    "Represent this sentence for searching relevant passages:",
    use_fp16=True,
)

print("Embedding model loaded.")

# --------------------------------------------------
# CONNECT TO QDRANT
# --------------------------------------------------

client = QdrantClient(
    url="https://28766b4d-63d9-449b-bd9f-92a531436109.eu-west-2-0.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("Connected to Qdrant.")

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
):
    """
    Dense Retrieval

    Query
        ↓
    Embedding
        ↓
    Vector Search
        ↓
    Top-K Chunks
    """

    query_embedding = model.encode(
        query
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=k,
    )

    return results.points

# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    count = client.count(
        collection_name=COLLECTION_NAME
    )

    print(
        f"Collection contains "
        f"{count.count} vectors."
    )

    query = "LLM int8?"

    results = retrieve(
        query=query,
        k=5,
    )

    print("\n" + "=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nRank #{rank}"
        )

        print(
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result.payload.get('chunk_id')}"
        )

        print(
            f"Paper: "
            f"{result.payload.get('paper_name')}"
        )

        print(
            f"Page: "
            f"{result.payload.get('page')}"
        )

        print(
            "\nText:\n"
            f"{result.payload.get('text')[:500]}"
        )

        print("-" * 80)