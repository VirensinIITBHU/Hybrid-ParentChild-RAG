"""
create_qa_dataset.py

Chunk-Level QA Dataset Generation for RAG Retrieval Evaluation

Improvements over v1:
- Target QA count enforced in code, not in the LLM prompt
- Question-type tagging (concept / mechanism / motivation / tradeoff / limitation)
- Cross-chunk deduplication via semantic similarity (TF-IDF cosine)
- Diversity balancing across papers
- Checkpoint saving — resume after a crash
- Exponential backoff retry on API failures
- Fixed is_good_chunk (case-insensitive patterns, tighter heuristics)
- Ground-truth containment check (answer substring must appear in chunk)
- Rich progress reporting via tqdm
"""

import json
import os
import random
import time
import hashlib
from pathlib import Path
from typing import List, Optional

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ingest import load_chunks

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────

load_dotenv()

OUTPUT_PATH      = Path("data/evaluation/qa_set.json")
CHECKPOINT_PATH  = Path("data/evaluation/qa_set_checkpoint.json")

TARGET_QA        = 50          # enforced in code
MAX_CHUNKS       = 200
QUESTIONS_PER_CHUNK = 2        # upper bound sent to LLM
RANDOM_SEED      = 42

DEDUP_THRESHOLD  = 0.85        # cosine sim above this → duplicate
MAX_PER_PAPER    = 10          # diversity cap per paper title

MAX_RETRIES      = 3
RETRY_BACKOFF    = 2.0         # seconds, doubles each retry

QUESTION_TYPES = [
    "concept",
    "mechanism",
    "motivation",
    "tradeoff",
    "limitation",
]

# ──────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────

class GeneratedQA(BaseModel):
    question:     str
    ground_truth: str
    question_type: str  # one of QUESTION_TYPES

    @field_validator("question_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in QUESTION_TYPES:
            return "concept"   # safe fallback, never raise
        return v

    @field_validator("question", "ground_truth")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty.")
        return v


class GeneratedChunkQA(BaseModel):
    samples: List[GeneratedQA]


class QASample(BaseModel):
    question:      str
    ground_truth:  str
    question_type: str
    paper_title:   str
    chunk_id:      int
    parent_id:     int
    chunk_text:    str
    chunk_hash:    str   # sha256[:12] — stable chunk identifier


# ──────────────────────────────────────────────────
# CLIENT
# ──────────────────────────────────────────────────

client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
)

# ──────────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are building a retrieval benchmark for evaluating RAG systems on research papers.

Generate question-answer pairs that test whether a retriever can locate the exact chunk
that answers the question. A good benchmark question is one where the answer is uniquely
anchored to this chunk and could not be answered from general knowledge alone.

Rules
─────
1. Generate between 0 and {QUESTIONS_PER_CHUNK} pairs per chunk.
2. Skip the chunk entirely (return empty list) if it contains:
   • OCR noise or garbled text
   • reference lists, bibliography, or footnotes
   • figure/table captions or raw table data
   • isolated numbers or scores without context
   • prompt demonstrations or synthetic examples
   • incomplete sentences or fragments

3. Never use outside knowledge. Never infer missing information.
4. The ground_truth must be a complete sentence drawn directly from the chunk text.
5. Prefer questions about: concepts, mechanisms, motivations, architectures, tradeoffs, limitations.
6. Avoid questions whose answers depend primarily on: exact benchmark numbers, figure/table
   references, section numbers, or page numbers.
7. For each pair, set question_type to exactly one of:
   concept | mechanism | motivation | tradeoff | limitation
8. Do not generate paraphrase duplicates within the same response.
9. Return valid JSON only — no preamble, no markdown fences.

Output schema (array of objects):
[
  {{
    "question":      "<string>",
    "ground_truth":  "<string>",
    "question_type": "<concept|mechanism|motivation|tradeoff|limitation>"
  }}
]
""".strip()

# ──────────────────────────────────────────────────
# CHUNK FILTERING
# ──────────────────────────────────────────────────

# All patterns lowercased — compared against text.lower()
_BAD_PATTERNS = [
    "references", "bibliography", "appendix",
    "acknowledgement", "acknowledgments",
    "q:", "question:", "answer:", "prompt:", "example:",
    "table ", "figure ", "fig.", "tab.",
    "\t",           # TSV / raw table dumps
]

_MIN_WORDS = 40


def is_good_chunk(text: str) -> bool:
    if len(text.split()) < _MIN_WORDS:
        return False
    lower = text.lower()
    return not any(pat in lower for pat in _BAD_PATTERNS)


# ──────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────

class DedupIndex:
    """TF-IDF cosine deduplication over accumulated questions."""

    def __init__(self, threshold: float = DEDUP_THRESHOLD):
        self.threshold = threshold
        self.questions: List[str] = []

    def is_duplicate(self, question: str) -> bool:
        if not self.questions:
            return False
        corpus = self.questions + [question]
        vec = TfidfVectorizer().fit_transform(corpus)
        sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
        return float(sims.max()) >= self.threshold

    def add(self, question: str) -> None:
        self.questions.append(question)


# ──────────────────────────────────────────────────
# GROUND-TRUTH CONTAINMENT CHECK
# ──────────────────────────────────────────────────

def _key_phrase(text: str, n: int = 6) -> str:
    """Return the first n words of text, lowercased."""
    return " ".join(text.lower().split()[:n])


def ground_truth_in_chunk(ground_truth: str, chunk: str) -> bool:
    """Check that a key phrase from the answer appears in the chunk."""
    phrase = _key_phrase(ground_truth)
    return phrase in chunk.lower()


# ──────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────

def load_checkpoint() -> List[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data['samples'])} samples loaded.")
        return data["samples"]
    return []


def save_checkpoint(dataset: List[dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"samples": dataset}, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────
# LLM CALL WITH RETRY
# ──────────────────────────────────────────────────

def call_llm_with_retry(chunk_text: str) -> Optional[GeneratedChunkQA]:
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model="openai/gpt-4o-mini",
                response_model=GeneratedChunkQA,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": chunk_text},
                ],
            )
        except Exception as e:
            if attempt == MAX_RETRIES:
                return None
            print(f"  ↺ Retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(delay)
            delay *= 2
    return None


# ──────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ──────────────────────────────────────────────────

def generate_dataset() -> None:
    print("Loading chunks...")
    child_docs, _ = load_chunks()
    print(f"Loaded {len(child_docs)} chunks.")

    random.seed(RANDOM_SEED)
    sampled_docs = random.sample(child_docs, min(MAX_CHUNKS, len(child_docs)))

    # Resume from checkpoint if available
    dataset = load_checkpoint()
    seen_chunk_hashes = {s["chunk_hash"] for s in dataset}

    dedup = DedupIndex()
    for s in dataset:
        dedup.add(s["question"])

    paper_counts: dict[str, int] = {}
    for s in dataset:
        paper_counts[s["paper_title"]] = paper_counts.get(s["paper_title"], 0) + 1

    progress = tqdm(sampled_docs, desc="Generating QA", unit="chunk")

    for doc in progress:
        if len(dataset) >= TARGET_QA:
            break

        chunk_text  = doc.page_content
        chunk_hash  = hashlib.sha256(chunk_text.encode()).hexdigest()[:12]
        paper_title = doc.metadata.get("paper_name", "unknown")
        chunk_id    = doc.metadata.get("chunk_id", -1)
        parent_id   = doc.metadata.get("parent_id", -1)

        # Skip already-processed chunks (checkpoint)
        if chunk_hash in seen_chunk_hashes:
            continue

        if not is_good_chunk(chunk_text):
            continue

        # Diversity cap
        if paper_counts.get(paper_title, 0) >= MAX_PER_PAPER:
            continue

        progress.set_postfix(paper=paper_title[:30], total=len(dataset))

        result = call_llm_with_retry(chunk_text)
        if result is None:
            print(f"  ✗ Skipping chunk {chunk_id} after {MAX_RETRIES} retries.")
            continue

        seen_chunk_hashes.add(chunk_hash)

        for qa in result.samples:
            if len(dataset) >= TARGET_QA:
                break

            # Ground-truth containment guard
            if not ground_truth_in_chunk(qa.ground_truth, chunk_text):
                continue

            # Cross-chunk deduplication
            if dedup.is_duplicate(qa.question):
                continue

            sample = QASample(
                question=qa.question,
                ground_truth=qa.ground_truth,
                question_type=qa.question_type,
                paper_title=paper_title,
                chunk_id=chunk_id,
                parent_id=parent_id,
                chunk_text=chunk_text,
                chunk_hash=chunk_hash,
            )
            dataset.append(sample.model_dump())
            dedup.add(qa.question)
            paper_counts[paper_title] = paper_counts.get(paper_title, 0) + 1

        # Save progress after every chunk
        save_checkpoint(dataset)

    # ── Final output ──────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"samples": dataset}, f, indent=4, ensure_ascii=False)

    # Clean up checkpoint on success
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # ── Summary ───────────────────────────────────
    type_counts = {}
    for s in dataset:
        t = s["question_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    paper_dist = {}
    for s in dataset:
        p = s["paper_title"]
        paper_dist[p] = paper_dist.get(p, 0) + 1

    print("\n══════════════════════════════════════")
    print(f"  Saved {len(dataset)} / {TARGET_QA} target samples")
    print(f"  Output → {OUTPUT_PATH}")
    print("\n  Question-type breakdown:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<12} {c}")
    print("\n  Paper distribution:")
    for p, c in sorted(paper_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c:>3}  {p}")
    print("══════════════════════════════════════\n")


# ──────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset()