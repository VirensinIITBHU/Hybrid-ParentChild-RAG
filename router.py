import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Dedicated client for routing so it never has to touch generate.py
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

CHAT_PATTERNS = [
    "are you sure",
    "really",
    "why",
    "how so",
    "explain more",
    "elaborate",
    "tell me more",
    "can you explain",
    "i don't understand",
    "what do you mean",
]

FOLLOWUP_WORDS = {"it", "this", "that", "they", "them", "its", "their"}


def route_query(query):
    query_lower = query.lower().strip()
    for phrase in CHAT_PATTERNS:
        if phrase in query_lower:
            return "CHAT"

    tokens = query_lower.split()
    if len(tokens) <= 3:
        return "AMBIGUOUS"

    return "RETRIEVE"


def llm_route(query, history):
    prompt = f"""
You are a routing classifier for a Research Paper RAG system.
Choose exactly one: CHAT or RETRIEVE

CHAT:
- Small talk, greetings ("hi", "hello", "how are you")
- Direct follow-up corrections ("are you sure?", "really?")
- Requests to change formatting ("can you summarize that shorter?")

RETRIEVE:
- Factual questions about concepts ("what is attention", "explain self-attention")
- Questions about papers, authors, data, or technical definitions
- Anything requiring knowledge from documents

Conversation History:
{history}

User Query:
{query}

Return ONLY the word CHAT or RETRIEVE. No punctuation or extra text.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    decision = response.choices[0].message.content.strip().upper()
    return decision if decision in ["CHAT", "RETRIEVE"] else "RETRIEVE"