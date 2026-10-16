
import pandas as pd
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting the paths
CHROMA_DIR = "chroma_store"
COLLECTION = "reset_vs_rboot_router"
TXT_PATH = os.path.join("data", "reset_vs_rboot_router.txt")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_reset_vs_router(path:str)-> list[Document]:
    file_path = Path(TXT_PATH)
    doc=[]
    text = file_path.read_text(encoding="utf-8")

    doc.append(Document(
        page_content= text,
        metadata= {
        "source": file_path.name,
        "category": "reset_vs_router"
        },
    ))
    return doc
def main():
    print("Loading the documents")
    
    docs = load_reset_vs_router(TXT_PATH)
    print(f"{len(docs)} run speed test entries loaded.")
    
    print("Initializing the embedding model")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print("creating vector store")
    vectorstore=Chroma.from_documents(
        persist_directory=CHROMA_DIR,
        embedding=embeddings,
        collection_name=COLLECTION,
        documents=docs,

    )
    print(
            f"Done. {vectorstore._collection.count()} vectors stored."
    )
if __name__=="__main__":
    main()

#print(document)

