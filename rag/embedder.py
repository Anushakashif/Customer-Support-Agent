import os
import logging
import faiss
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY
)

# ── Text splitter ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ── Where FAISS index is saved locally ───────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index"

# ── Sample knowledge base ─────────────────────────────────────────────────────
SAMPLE_DOCS = [
    "To reset your password, go to Settings > Security > Reset Password. You will receive an email within 2 minutes.",
    "Refunds are processed within 5-7 business days back to your original payment method. Contact billing@support.com for help.",
    "If your payment failed, check that your card details are correct and your billing address matches your bank records.",
    "To cancel your subscription, go to Settings > Billing > Cancel Plan. You will retain access until the end of your billing period.",
    "For technical issues, try clearing your browser cache first. If the problem persists, contact our technical team.",
    "Our service is available 24/7. Planned maintenance windows are announced 48 hours in advance on our status page.",
    "To upgrade your plan, go to Settings > Billing > Upgrade. Changes take effect immediately and are prorated.",
    "If you see error code 404, the resource you are looking for does not exist. Check the URL and try again.",
    "If you see error code 500, there is a server issue on our end. Please wait 10 minutes and try again.",
    "To add a team member, go to Settings > Team > Invite Member. They will receive an email invitation.",
]

# ── Load and save knowledge base ──────────────────────────────────────────────
def load_knowledge_base() -> FAISS:
    """
    Splits docs into chunks, embeds them,
    saves FAISS index to disk, returns the vector store.
    """
    logger.info("Loading knowledge base into FAISS...")

    chunks = splitter.create_documents(SAMPLE_DOCS)
    logger.info(f"Created {len(chunks)} chunks from {len(SAMPLE_DOCS)} documents")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save to disk so we don't re-embed every time
    vector_store.save_local(FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to '{FAISS_INDEX_PATH}'")

    return vector_store


# ── Load existing index from disk ─────────────────────────────────────────────
def get_vector_store() -> FAISS:
    """
    Loads the saved FAISS index from disk.
    Runs load_knowledge_base() first if index doesn't exist.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning("FAISS index not found, rebuilding...")
        return load_knowledge_base()

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # safe since we created this file ourselves
    )