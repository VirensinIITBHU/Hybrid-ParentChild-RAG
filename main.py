import argparse


# --------------------------------------------------
# INGEST
# --------------------------------------------------

def run_ingest():
    from ingest import main
    main()


# --------------------------------------------------
# EMBED
# --------------------------------------------------

def run_embed():
    import embed


# --------------------------------------------------
# CHAT
# --------------------------------------------------

def run_chat():
    from generate import generate_answer

    print("\nFirstRAG Chat")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nYou: ")

        if query.lower().strip() == "exit":
            break

        answer = generate_answer(query)

        print(f"\nAssistant:\n{answer}")


# --------------------------------------------------
# DENSE RETRIEVAL TEST
# --------------------------------------------------

def run_dense():

    from retrieve import retrieve

    query = input("\nQuery: ")

    results = retrieve(query, k=5)

    print("\nDense Retrieval Results\n")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"Score: {result.score:.4f}")
        print(
            f"Paper: "
            f"{result.payload.get('paper_name')}"
        )
        print(
            result.payload.get("text", "")[:400]
        )


# --------------------------------------------------
# BM25 TEST
# --------------------------------------------------

def run_bm25():

    from retrieve_bm25 import bm25_retrieve

    query = input("\nQuery: ")

    results = bm25_retrieve(
        query=query,
        k=5,
    )

    print("\nBM25 Results\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(f"\nRank {rank}")
        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Paper: "
            f"{result['document'].metadata['paper_name']}"
        )

        print(
            result["document"].page_content[:400]
        )


# --------------------------------------------------
# HYBRID TEST
# --------------------------------------------------

def run_hybrid():

    from retrieve_reranked_hybrid import (
        retrieve_hybrid,
    )

    query = input("\nQuery: ")

    results = retrieve_hybrid(
        query=query,
        k=5,
    )

    print("\nHybrid Results\n")

    for rank, (
        parent_id,
        parent_doc,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        print(f"\nRank {rank}")

        print(
            f"Paper: "
            f"{parent_doc.metadata.get('paper_name')}"
        )

        print(
            f"Page: "
            f"{parent_doc.metadata.get('page')}"
        )

        print(
            f"Reranker Score: "
            f"{score:.4f}"
        )

        print(
            parent_doc.page_content[:500]
        )


# --------------------------------------------------
# ROUTER TEST
# --------------------------------------------------

def run_router():

    from router import (
        route_query,
        llm_route,
    )

    query = input("\nQuery: ")

    route = route_query(query)

    if route == "AMBIGUOUS":
        route = llm_route(query, "")

    print(
        f"\nFinal Route: {route}"
    )


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
#for evaluation, see evaluate_hybrid.py and evaluate_own_ragas_v2.py
#TO run other evcalutions, run them separately as they may require different setups and are not fully integrated into this control center yet or not imp in final pipeline :))
def run_eval():

    from evaluate_hybrid import main

    main()


def run_ragas():

    from evaluate_own_ragas_v2 import main

    main()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="FirstRAG Control Center"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "ingest",
            "embed",
            "chat",
            "dense",
            "bm25",
            "hybrid",
            "router",
            "eval",
            "ragas",
        ],
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest()

    elif args.mode == "embed":
        run_embed()

    elif args.mode == "chat":
        run_chat()

    elif args.mode == "dense":
        run_dense()

    elif args.mode == "bm25":
        run_bm25()

    elif args.mode == "hybrid":
        run_hybrid()

    elif args.mode == "router":
        run_router()

    elif args.mode == "eval":
        run_eval()

    elif args.mode == "ragas":
        run_ragas()


if __name__ == "__main__":
    main()