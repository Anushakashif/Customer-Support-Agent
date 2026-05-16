import logging
from langchain_community.vectorstores import FAISS
from rag.embedder import get_vector_store

logger = logging.getLogger(__name__)

def retrieve_relevant_docs(query: str, k: int = 3) -> list[str]:
    """
    Searches FAISS index for top-k most relevant
    chunks for the given query.
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=k)

        docs = [doc.page_content for doc in results]
        logger.info(f"Retrieved {len(docs)} docs for query: '{query[:50]}'")
        return docs

    except Exception as e:
        logger.error(f"Retriever error: {e}")
        return []