
import pandas as pd
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setting the paths
CHROMA_DIR = "chroma_store"
COLLECTION = "run_speed_test"
TXT_PATH = os.path.join("data", "run_speed_test.txt")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_run_speed(path:str)-> list[Document]:
    file_path = Path(TXT_PATH)
    doc=[]
    text = file_path.read_text(encoding="utf-8")

    doc.append(Document(
        page_content= text,
        metadata= {
        "source": file_path.name,
        "category": "run_speed_test"
        },
    ))
    return doc
def main():
    print("Loading the documents")
    
    docs = load_run_speed(TXT_PATH)
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

