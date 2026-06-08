
"""
create_qa_dataset.py

Chunk-Level QA Dataset Generation
for Retrieval Evaluation
"""

import json
import os
import random
from typing import List

import instructor

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ingest import load_chunks

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

OUTPUT_PATH = (
    "data/evaluation/qa_set.json"
)

MAX_CHUNKS = 200

QUESTIONS_PER_CHUNK = 2

RANDOM_SEED = 42

# --------------------------------------------------
# PYDANTIC
# --------------------------------------------------

class GeneratedQA(BaseModel):

    question: str

    ground_truth: str


class GeneratedChunkQA(BaseModel):

    samples: List[GeneratedQA]


class QASample(BaseModel):

    question: str

    ground_truth: str

    paper_title: str

    chunk_id: int

    parent_id: int

    chunk_text: str


# --------------------------------------------------
# CLIENT
# --------------------------------------------------

client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv(
            "GROQ_API_KEY"
        ),
        base_url=
        "https://api.groq.com/openai/v1",
    )
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = f"""
You are creating a retrieval benchmark for evaluating Retrieval-Augmented Generation (RAG) systems on research papers.

Your goal is to generate HIGH-QUALITY retrieval questions.

The benchmark will be used to evaluate whether a retriever can find the correct chunk.

IMPORTANT:

The final dataset must contain exactly **50 high-quality question-answer pairs in total** across all processed chunks.

A question-answer pair should only be generated when the answer is explicitly contained in the provided chunk.

Rules:

1. Generate between 0 and {QUESTIONS_PER_CHUNK} question-answer pairs for each chunk.

2. Across all chunks, the final dataset must contain exactly 50 question-answer pairs.

3. If the chunk contains:

   * OCR noise
   * references
   * bibliography entries
   * figure captions
   * table dumps
   * isolated numerical values
   * prompt examples
   * incomplete information
   * synthetic demonstrations

   then return an empty list.

4. NEVER use outside knowledge.

5. NEVER infer missing information.

6. If the answer is not explicitly stated in the chunk, do not generate a question.

7. Prefer questions about:

   * concepts
   * motivations
   * mechanisms
   * architectures
   * methods
   * limitations
   * tradeoffs

8. Avoid questions that depend primarily on:

   * exact numbers
   * benchmark scores
   * figure numbers
   * table rows
   * page numbers
   * section numbers

9. Questions should resemble what a researcher, student, or engineer would naturally ask after reading the chunk.

10. Answers must:

* be directly supported by the chunk
* be complete sentences
* not introduce new information

11. Avoid paraphrase duplicates.
    Do not ask multiple questions whose answers are essentially identical.

12. Prefer fewer high-quality questions over low-quality questions for any individual chunk.

13. Return JSON only.

Example:

Chunk:
"Switch-Base is more sample efficient and yields a 2.5x speedup."

Output:
[

"question": "What advantage does Switch-Base provide compared to dense models?",
"ground-truth": "Switch-Base is more sample efficient and yields a 2.5x speedup."
... and so on acc to schema

]

Example:

Chunk:
"Q: What is the capital of California?
A:"

Output:
[]

Example:

Chunk:
"0.65 0.70 1.10 2.54 15.4"

Output:
[]

"""

# --------------------------------------------------
# CHUNK FILTER
# --------------------------------------------------

def is_good_chunk(
    text: str,
) -> bool:

    words = len(
        text.split()
    )

    if words < 80:
        return False

    lower = text.lower()

    bad_patterns = [

        "references",

        "bibliography",

        "appendix",

        "acknowledgement",

        "Q:",
        "Question:",
        "Answer:",
        "Prompt:",
        "Example:",
        "Table",
        "Figure",

    ]

    for pattern in bad_patterns:

        if pattern in lower:
            return False

    return True


# --------------------------------------------------
# GENERATE
# --------------------------------------------------

def generate_dataset():

    print(
        "Loading chunks..."
    )

    child_docs, _ = load_chunks()

    print(
        f"Loaded "
        f"{len(child_docs)} chunks."
    )

    random.seed(
        RANDOM_SEED
    )

    sampled_docs = random.sample(
        child_docs,
        min(
            MAX_CHUNKS,
            len(child_docs)
        )
    )

    dataset = []

    for doc in sampled_docs:

        chunk_text = (
            doc.page_content
        )

        if not is_good_chunk(
            chunk_text
        ):
            continue

        paper_title = (
            doc.metadata.get(
                "paper_name",
                "unknown"
            )
        )

        chunk_id = (
            doc.metadata.get(
                "chunk_id",
                -1
            )
        )

        parent_id = (
        doc.metadata.get(
            "parent_id",
            -1
        )
    )

        print(
            f"Generating QA | "
            f"{paper_title}"
        )

        try:

            generated = (
                client.chat.completions.create(

                    model=
                    "llama-3.3-70b-versatile",

                    response_model=
                    GeneratedChunkQA,

                    messages=[

                        {
                            "role":
                            "system",

                            "content":
                            SYSTEM_PROMPT,
                        },

                        {
                            "role":
                            "user",

                            "content":
                            chunk_text,
                        },
                    ],
                )
            )

            for qa in generated.samples:

                sample = QASample(

                question=
                qa.question,

                ground_truth=
                qa.ground_truth,

                paper_title=
                paper_title,

                chunk_id=
                chunk_id,

                parent_id=
                parent_id,

                chunk_text=
                chunk_text,
            )
                dataset.append(
                    sample.model_dump()
                )

        except Exception as e:

            print(
                f"Failed chunk "
                f"{chunk_id}: {e}"
            )

    os.makedirs(
        "data/evaluation",
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "samples":
                dataset
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n=================="
    )

    print(
        f"Saved "
        f"{len(dataset)} "
        f"samples"
    )

    print(
        f"Output -> "
        f"{OUTPUT_PATH}"
    )

    print(
        "=================="
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    generate_dataset()