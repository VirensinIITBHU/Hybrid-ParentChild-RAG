"""
embed.py

Purpose:
---------
Convert text chunks into dense vector embeddings
and store them inside Qdrant.

Pipeline:

child_chunks.pkl
        ↓
Embedding Model
        ↓
Dense Vectors
        ↓
Qdrant
        ↓
Semantic Retrieval

This file is responsible for building the vector index.
Without this step retrieval cannot work.
"""

import os

from dotenv import load_dotenv

from ingest import load_chunks

from FlagEmbedding import FlagAutoModel

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

load_dotenv()

COLLECTION_NAME = "research_papers"

BATCH_SIZE = 100

# ---------------------------------------------------
# LOAD CHUNKS
# ---------------------------------------------------

print("\nLoading chunks...")

child_docs, parent_docs = load_chunks()

print(
    f"Loaded {len(child_docs)} child chunks."
)

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

print("\nLoading embedding model...")

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-small-en-v1.5",
    query_instruction_for_retrieval=
    "Represent this sentence for searching relevant passages:",
    use_fp16=True,
)

print("Embedding model loaded.")

# ---------------------------------------------------
# DETECT EMBEDDING DIMENSION
# ---------------------------------------------------

sample_vector = model.encode(
    ["hello world"]
)[0]

VECTOR_SIZE = len(sample_vector)

print(
    f"Embedding dimension: {VECTOR_SIZE}"
)

# ---------------------------------------------------
# CONNECT TO QDRANT
# ---------------------------------------------------

print("\nConnecting to Qdrant...")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv(
        "QDRANT_API_KEY"
    ),
)
    

print("Connected to Qdrant.")
    

# ---------------------------------------------------
# RECREATE COLLECTION
# ---------------------------------------------------

existing = [
    c.name
    for c in client.get_collections().collections
]

if COLLECTION_NAME in existing:

    print(
        f"Deleting existing collection "
        f"'{COLLECTION_NAME}'..."
    )

    client.delete_collection(
        COLLECTION_NAME
    )

print(
    f"Creating collection "
    f"'{COLLECTION_NAME}'..."
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)

print("Collection created.")

# ---------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------

print("\nGenerating embeddings...")

texts = [
    doc.page_content
    for doc in child_docs
]

embeddings = model.encode(texts)

print(
    f"Generated {len(embeddings)} vectors."
)

# ---------------------------------------------------
# BUILD QDRANT POINTS
# ---------------------------------------------------

points = []

for idx, (doc, embedding) in enumerate(
    zip(child_docs, embeddings)
):

    points.append(
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={

            "text": doc.page_content,

            "paper_name":
            doc.metadata.get(
                "paper_name"
            ),

            "source":
            doc.metadata.get(
                "source"
            ),

            "page":
            doc.metadata.get(
                "page"
            ),

            "chunk_id":
            doc.metadata.get(
                "chunk_id"
            ),

            "parent_id":
            doc.metadata.get(
                "parent_id"
            ),

            "chunk_type":
            doc.metadata.get(
                "chunk_type"
            ),
        },
        )
    )

# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------

print("\nUploading vectors...")

for start in range(
    0,
    len(points),
    BATCH_SIZE,
):

    batch = points[
        start:
        start + BATCH_SIZE
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
        wait=True,
    )

    print(
        f"Uploaded "
        f"{start + len(batch)}"
        f"/{len(points)}"
    )

# ---------------------------------------------------
# VERIFY
# ---------------------------------------------------

count = client.count(
    collection_name=COLLECTION_NAME
)

print("\nIndexing Complete")

print(
    f"Total vectors stored: "
    f"{count.count}"
)