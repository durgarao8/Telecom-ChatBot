
import pandas as pd
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setting the paths
CHROMA_DIR = "chroma_store"
COLLECTION = "faq"
CSV_PATH = os.path.join("data", "faq.csv")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []

    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "faq",
                    "category": row["category"],
                    "faq_id": str(row["id"]),
                },
            )
        )

    return docs


def main():
    print("Loading the documents")

    docs = load_faq_documents(CSV_PATH)
    print(f"{len(docs)} FAQ entries loaded.")

    print("Initializing the embedding model")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print("Creating vector store")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )

    print(
        f"Done. {vectorstore._collection.count()} vectors stored."
    )


if __name__ == "__main__":
    main()
    