
"""
Hybrid retriever with cross-encoder reranking.

Sources:
- FAQ Chroma collection
- Tickets Chroma collection
- Guides Chroma collection
- BM25 keyword search across all stored documents
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder


CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"

K_FAQ = 10
K_TICKETS = 10
K_GUIDES = 10
K_BM25 = 10
TOP_K = 5
K_RUN_SPEED=10
K_RESTART_ROUTER=10
K_RESET_VS_ROUTER=10


def build_retriever() -> RunnableLambda:
    print("Initializing embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    # Connect to existing Chroma collections
    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    guide_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    run_speed_store=Chroma(
        collection_name="run_speed_test",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    restart_router_store=Chroma(
            collection_name="restart_router",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    reset_vs_rboot_router_store=Chroma(
            collection_name="reset_vs_rboot_router",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    # Vector retrievers
    faq_retriever = faq_store.as_retriever(
        search_kwargs={"k": K_FAQ}
    )

    tickets_retriever = tickets_store.as_retriever(
        search_kwargs={"k": K_TICKETS}
    )

    guides_retriever = guide_store.as_retriever(
        search_kwargs={"k": K_GUIDES}
    )

    run_speed_retriever=run_speed_store.as_retriever(
        search_kwargs={"k":K_RUN_SPEED}
    )
    restart_router_retriever=restart_router_store.as_retriever(
            search_kwargs={"k":K_RESTART_ROUTER}
        )
    reset_vs_rboot_router_retriever=reset_vs_rboot_router_store.as_retriever(
            search_kwargs={"k":K_RESET_VS_ROUTER}
        )

    # Load stored documents for the BM25 index
    data_chunks = []

    for store in (faq_store, tickets_store, guide_store,run_speed_store,reset_vs_rboot_router_store,restart_router_store):
        stored = store.get(
            include=["documents", "metadatas"]
        )

        contents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        for content, metadata in zip(contents, metadatas):
            if content:
                data_chunks.append(
                    Document(
                        page_content=content,
                        metadata=metadata or {},
                    )
                )

    # Remove duplicate documents from the BM25 corpus
    unique_chunks = []
    seen = set()

    for doc in data_chunks:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("faq_id", ""),
            doc.metadata.get("ticket_id", ""),
            doc.metadata.get("chunk_index", ""),
            doc.page_content,
        )

        if key not in seen:
            seen.add(key)
            unique_chunks.append(doc)

    if not unique_chunks:
        raise ValueError(
            "No documents found in Chroma. "
            "Populate the FAQ, tickets, and guides collections first."
        )

    print(f"Loaded {len(unique_chunks)} documents for BM25.")

    keyword_retriever = BM25Retriever.from_documents(
        unique_chunks
    )
    keyword_retriever.k = K_BM25

    # Hybrid retrieval using reciprocal-rank fusion
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            keyword_retriever,
            faq_retriever,
            tickets_retriever,
            guides_retriever,
            run_speed_retriever,
            reset_vs_rboot_router_retriever,
            restart_router_retriever
        ],
        weights = [0.20, 0.25, 0.20, 0.25, 0.03, 0.035, 0.035],
    )

    print("Loading cross-encoder reranker...")

    reranker = CrossEncoder(RERANK_MODEL)

    def retrieve_and_rerank(query: str) -> list[Document]:
        # 1. Retrieve candidate documents
        candidate_docs = hybrid_retriever.invoke(query)

        # 2. Remove duplicate candidates
        unique_docs = []
        seen = set()

        for doc in candidate_docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("faq_id", ""),
                doc.metadata.get("ticket_id", ""),
                doc.metadata.get("chunk_index", ""),
                doc.page_content,
            )

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        if not unique_docs:
            return []

        # 3. Score query-document pairs
        pairs = [
    [
        query,
        (
            f"Document: {doc.metadata.get('source', '')}\n"
            f"Category: {doc.metadata.get('category', '')}\n"
            f"Content: {doc.page_content}"
        )
        ]
        for doc in unique_docs
    ]
        scores = reranker.predict(pairs)

        # 4. Rank by cross-encoder score
        ranked_docs = sorted(
            zip(unique_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # 5. Return the top-ranked Documents
        results = []

        for doc, score in ranked_docs[:TOP_K]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)

        return results

    return RunnableLambda(retrieve_and_rerank)
