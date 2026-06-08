"""
ingest.py

Purpose:
--------
This file is responsible for:

1. Loading research papers (PDFs)
2. Splitting them into chunks
3. Enriching chunks with metadata
4. Saving chunks locally for later embedding/indexing

Pipeline:

PDFs
 ↓
LangChain Documents
 ↓
Chunking
 ↓
Metadata Enrichment
 ↓
Pickle Storage
 ↓
embed.py / retrieve.py

Why Chunking?

LLMs and embedding models have context limits.

Instead of embedding entire papers, we split them into
smaller semantic units called chunks.

Example:

Paper:
----------------------------------
Attention Is All You Need
(15 pages)
----------------------------------

Becomes:

Chunk 1
Chunk 2
Chunk 3
...
Chunk N

Each chunk can then be embedded independently.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.documents import Document


# Load environment variables
load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

PDF_DIRECTORY = "documents/AI research Papers"

# Small chunks for retrieval
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 80

# Larger chunks for parent retrieval
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

CHUNK_SAVE_DIR = "data/chunks"


# ------------------------------------------------------------------
# LOAD DOCUMENTS
# ------------------------------------------------------------------

def load_documents() -> List[Document]:
    """
    Loads all PDFs from the configured directory.

    Returns:
        List[Document]

    Raises:
        FileNotFoundError:
            If PDF directory doesn't exist.

        ValueError:
            If no PDFs are found.
    """

    if not os.path.exists(PDF_DIRECTORY):
        raise FileNotFoundError(
            f"Directory '{PDF_DIRECTORY}' does not exist."
        )

    loader = DirectoryLoader(
        PDF_DIRECTORY,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            f"No PDF documents found in '{PDF_DIRECTORY}'."
        )

    print(f"\nLoaded {len(documents)} pages.")

    return documents


# ------------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------------

def create_chunks(
    documents: List[Document]
) -> Tuple[List[Document], List[Document]]:

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    parent_docs = parent_splitter.split_documents(
        documents
    )

    child_docs = []

    child_id = 0

    for parent_id, parent_doc in enumerate(
        parent_docs
    ):

        child_chunks = child_splitter.split_documents(
            [parent_doc]
        )

        for child in child_chunks:

            child.metadata["parent_id"] = (
                parent_id
            )

            child.metadata["chunk_id"] = (
                child_id
            )

            child_docs.append(
                child
            )

            child_id += 1

    return child_docs, parent_docs


# ------------------------------------------------------------------
# METADATA ENRICHMENT
# ------------------------------------------------------------------

def enrich_metadata(
    child_docs: List[Document],
    parent_docs: List[Document],
) -> Tuple[List[Document], List[Document]]:

    for doc in child_docs:

        filename = Path(
            doc.metadata.get(
                "source",
                "unknown"
            )
        ).stem

        doc.metadata["paper_name"] = (
            filename
        )

        doc.metadata["chunk_type"] = (
            "child"
        )

    for idx, doc in enumerate(
        parent_docs
    ):

        filename = Path(
            doc.metadata.get(
                "source",
                "unknown"
            )
        ).stem

        doc.metadata["paper_name"] = (
            filename
        )

        doc.metadata["chunk_type"] = (
            "parent"
        )

        doc.metadata["parent_id"] = (
            idx
        )

    return child_docs, parent_docs


# ------------------------------------------------------------------
# SAVE CHUNKS
# ------------------------------------------------------------------

def save_chunks(
    child_docs: List[Document],
    parent_docs: List[Document],
) -> None:
    """
    Stores chunked documents on disk.

    Files created:

    data/chunks/child_chunks.pkl
    data/chunks/parent_chunks.pkl
    """

    os.makedirs(
        CHUNK_SAVE_DIR,
        exist_ok=True,
    )

    child_path = (
        f"{CHUNK_SAVE_DIR}/child_chunks.pkl"
    )

    parent_path = (
        f"{CHUNK_SAVE_DIR}/parent_chunks.pkl"
    )

    with open(child_path, "wb") as f:
        pickle.dump(child_docs, f)

    with open(parent_path, "wb") as f:
        pickle.dump(parent_docs, f)

    print(f"\nSaved child chunks → {child_path}")
    print(f"Saved parent chunks → {parent_path}")


# ------------------------------------------------------------------
# LOAD SAVED CHUNKS
# ------------------------------------------------------------------

def load_chunks():
    """
    Utility function.

    Useful during embedding/indexing stage.
    """

    with open(
        f"{CHUNK_SAVE_DIR}/child_chunks.pkl",
        "rb",
    ) as f:
        child_docs = pickle.load(f)

    with open(
        f"{CHUNK_SAVE_DIR}/parent_chunks.pkl",
        "rb",
    ) as f:
        parent_docs = pickle.load(f)

    return child_docs, parent_docs


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():

    print("\nStarting ingestion pipeline...\n")

    documents = load_documents()

    child_docs, parent_docs = create_chunks(
        documents
    )

    child_docs, parent_docs = enrich_metadata(
        child_docs,
        parent_docs,
    )

    save_chunks(
        child_docs,
        parent_docs,
    )

    print("\n" + "=" * 60)

    print(
        f"Original Pages : {len(documents)}"
    )

    print(
        f"Child Chunks   : {len(child_docs)}"
    )

    print(
        f"Parent Chunks  : {len(parent_docs)}"
    )

    print("=" * 60)

    if child_docs:
        print("\nSample Metadata:")
        print(child_docs[0].metadata)


if __name__ == "__main__":
    main()