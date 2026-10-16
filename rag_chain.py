
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever
#import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a Telecom Customer Support Assistant.

Answer the user's question using only the information provided in the retrieved context.

Rules:
1. Use the retrieved context as your primary and only factual source.
2. Do not invent support phone numbers, app navigation paths, prices, policies, procedures, or technical details.
3. If the context does not contain the requested information, clearly say: "I couldn't find that information in the available telecom documentation."
4. Provide step-by-step instructions when the retrieved context supports them.
5. Distinguish clearly between rebooting a router and performing a factory reset.
6. Do not recommend a factory reset as the first troubleshooting step when a reboot may be sufficient.
7. Keep answers clear, concise, and relevant to the question.
8. Do not claim that an action was performed or that an issue was resolved unless the user confirms it.
9. If multiple retrieved documents conflict, do not guess; explain that the available information is inconsistent.

Retrieved context:
{context}

User question:
{question}  
"""


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents were found."

    sections = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()

        section = (
            f"[SOURCE: {source}]\n"
            f"{doc.page_content}"
        )

        sections.append(section)

    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not found. Check your .env file."
    )

    llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=4096,
    reasoning_format="parsed",
    timeout=None,
    )

    # Retrieve documents once and preserve them for evaluation.
    chain = (
        {
            "question": RunnablePassthrough(),
            "context_docs": retriever,
        }
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: _format_docs(x["context_docs"])
            )
        )
        | {
            "answer": prompt | llm | StrOutputParser(),
            "contexts": RunnableLambda(
                lambda x: [
                    doc.page_content for doc in x["context_docs"]
                ]
            ),
            "sources": RunnableLambda(
                lambda x: [
                    doc.metadata.get("source", "unknown")
                    for doc in x["context_docs"]
                ]
            ),
        }
    )

    return chain