# test.py

#```python
from retriever import build_retriever


def main():
    print("=" * 70)
    print("       TELECOM RAG RETRIEVER TEST")
    print("=" * 70)

    # Initialize hybrid retriever + cross-encoder reranker
    retriever = build_retriever()

    test_queries = [
        "How do I restart my router?",
        "What is the difference between rebooting and resetting a router?",
        "How can I check my internet speed?",
        "Why is my mobile internet not working?",
        "Why is my bill higher than usual?",
        "How do I activate international roaming?",
        "What is the difference between download speed and upload speed?",
        "My internet is slow. What should I do?",
    ]

    for i, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query}")
        print("=" * 70)

        try:
            # Invoke hybrid retrieval + reranking
            results = retriever.invoke(query)

            if not results:
                print("No documents retrieved.")
                continue

            for rank, doc in enumerate(results, start=1):
                print(f"\nRank: {rank}")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Category: {doc.metadata.get('category', 'Unknown')}")
                print(
                    "Rerank score:",
                    round(doc.metadata.get("rerank_score", 0.0), 4),
                )
                print("-" * 50)
                print(doc.page_content[:700])

        except Exception as exc:
            print(f"Error while testing query: {exc}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()