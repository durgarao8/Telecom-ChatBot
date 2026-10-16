
"""
CLI entry point for the telecom RAG chatbot.
Usage: python main.py
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from rag_chain import build_chain


def main():
    # Load environment variables from .env
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print(
            "Error: GROQ_API_KEY not found. "
            "Add it to your .env file."
        )
        return

    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")

    try:
        chain = build_chain()
    except Exception as exc:
        print(f"Error initializing the RAG chain: {exc}")
        return

    while True:
        try:
            question = input("Customer: ").strip()

            if not question:
                continue

            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            print("\nAssistant: ", end="", flush=True)

            try:
                for chunk in chain.stream(question):
                    print(chunk, end="", flush=True)

                print("\n")

            except Exception as exc:
                print(f"\nError generating response: {exc}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
