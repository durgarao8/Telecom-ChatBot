
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH = os.path.join("data", "telecom_guide.pdf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def main():
    print("Loading the PDF")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"{len(pages)} pages loaded")

    print(
        f"Chunking with size {CHUNK_SIZE} "
        f"and overlap {CHUNK_OVERLAP}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guides"
        chunk.metadata["chunk_index"] = i

    print(f"{len(chunks)} chunks produced")

    print("Initializing embedding model")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )

    print(
        f"Done. {vectorstore._collection.count()} vectors stored."
    )


if __name__ == "__main__":
    main()
