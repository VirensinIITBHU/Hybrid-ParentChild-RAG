import os
from dotenv import load_dotenv
from openai import OpenAI

from openai import OpenAI

# Notice we only import functions from router now, no circular loops
from router import route_query, llm_route
from retrieve_reranked_hybrid import retrieve_hybrid

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

chat_history = []
last_contexts = []


def chat_response(query):
    history = format_history()
    context = "\n\n".join(last_contexts)

    prompt = f"Conversation:\n{history}\n\nRelevant Context:\n{context}\n\nUser:\n{query}"
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def format_history():
    if not chat_history:
        return "No previous conversation."
    return "".join(f"{msg['role']}: {msg['content']}\n" for msg in chat_history[-6:])


def format_user_history():
    return "".join(msg["content"] + "\n" for msg in chat_history if msg["role"] == "user")


def rewrite_query(query):
    if not chat_history:
        return query

    AMBIGUOUS_WORDS = {"it", "this", "that", "they", "them", "its"}
    if not any(word in query.lower().split() for word in AMBIGUOUS_WORDS):
        return query

    history = format_user_history()
    prompt = f"Rewrite... \nHistory:\n{history}\nLatest:\n{query}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        standalone_query = response.choices[0].message.content.strip()
        print(f"\n[Rewritten Query] {standalone_query}")
        return standalone_query
    except Exception:
        return query


def build_context(retrieved_docs):
    return "\n\n".join(parent_doc.page_content for _, parent_doc, _ in retrieved_docs)


def generate_answer(query):
    global last_contexts
    history = format_history()
    route = route_query(query)

    if route == "AMBIGUOUS":
        route = llm_route(query, history)  # Passed client here!

    print(f"\n[Route] {route}")

    if route == "CHAT":
        answer = chat_response(query)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return answer

    # RETRIEVE ROUTE
    standalone_query = rewrite_query(query)
    docs = retrieve_hybrid(query=standalone_query, k=3)
    context = build_context(docs)

    last_contexts = [parent_doc.page_content for _, parent_doc, _ in docs]

    prompt = f""" You are a helpful AI assistant.

Instructions:
- First use the retrieved context as the primary source of truth.
- If the context is incomplete, use your general knowledge to supplement the answer.
- Do not contradict information found in the context.
- Explain concepts in clear and simple language.
- Synthesize information instead of copying sentences verbatim.
- If multiple context chunks contain relevant information, combine them into a single coherent answer.
- If the context is not relevant, answer using general knowledge and mention that the answer was not found in the retrieved documents.

Conversation History:
{history}

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    print("\nSources:")
    seen = set()
    for _, parent_doc, score in docs:
        paper = parent_doc.metadata.get("paper_name", "Unknown")
        page = parent_doc.metadata.get("page", "?")
        if (paper, page) in seen:
            continue
        seen.add((paper, page))
        print(f"{paper} | Page {page} | Score={score:.4f}")

    return answer

#experimented with k =5,2,3
#k = 3 is giving best results, k-5 (recall increase but precision decrease) k = 2 (vice versa)
def generate_answer_with_context(query):
    standalone_query = rewrite_query(query)
    docs = retrieve_hybrid(query=standalone_query, k=2)
    context = build_context(docs)
    history = format_history()

    prompt = f"Answer ONLY from context...\nHistory:\n{history}\nContext:\n{context}\nQuestion:\n{query}"
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    contexts = [parent_doc.page_content for _, parent_doc, _ in docs]

    # Cleaned up: code prints sources BEFORE returning the dictionary
    print("\nSources:")
    seen = set()
    for _, parent_doc, score in docs:
        paper = parent_doc.metadata.get("paper_name", "Unknown")
        page = parent_doc.metadata.get("page", "?")
        if (paper, page) in seen:
            continue
        seen.add((paper, page))
        print(f"{paper} | Page {page} | Score={score:.4f}")

    return {
        "answer": answer,
        "contexts": contexts,
        "retrieved_docs": docs,
    }


if __name__ == "__main__":
    print("\nFirstRAG Chat\nType 'exit' to quit.")
    while True:
        question = input("\nYou: ")
        if question.lower().strip() == "exit":
            print("\nGoodbye!")
            break
        answer = generate_answer(question)
        print(f"\nAssistant:\n{answer}")