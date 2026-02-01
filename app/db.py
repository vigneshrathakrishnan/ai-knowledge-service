import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ===============================
# Imports
# ===============================
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Token counting (optional)
try:
    import tiktoken
except ImportError:
    tiktoken = None

# ===============================
# Config
# ===============================
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_PRICE_PER_1K = 0.00002

# ===============================
# GLOBAL IN-MEMORY VECTOR STORE
# ===============================
_VECTOR_DB = None

# ===============================
# Utilities
# ===============================
def _estimate_tokens_and_cost(texts: List[str], model: str = EMBEDDING_MODEL_NAME) -> Dict[str, Any]:
    total_tokens = 0

    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        for t in texts:
            total_tokens += len(enc.encode(t))
    else:
        # rough estimate
        total_tokens = sum(len(t) for t in texts) // 4

    estimated_cost = (total_tokens / 1000.0) * EMBEDDING_PRICE_PER_1K
    return {
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 8)
    }

# ===============================
# MAIN FUNCTIONS
# ===============================
def create_vectorstore_from_data(documents: List[Document]) -> Dict[str, Any]:
    """
    Creates an IN-MEMORY Chroma vectorstore.
    This is DEMO-SAFE and avoids all SQLite / filesystem issues.
    """
    global _VECTOR_DB

    # ---- Chunking ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    chunk_texts = [c.page_content for c in chunks if c.page_content.strip()]

    estimate = _estimate_tokens_and_cost(chunk_texts)

    # ---- Build embeddings ----
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # ---- Create IN-MEMORY vector store ----
    _VECTOR_DB = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return {
        "chunks_created": len(chunks),
        "estimated_embedding_tokens": estimate["estimated_tokens"],
        "estimated_embedding_cost_usd": estimate["estimated_cost_usd"],
        "storage": "in-memory"
    }

def load_vectorstore():
    """
    Returns the active in-memory vectorstore.
    """
    return _VECTOR_DB

def debug_vectorstore() -> Dict[str, Any]:
    """
    Debug helper to confirm DB presence.
    """
    if _VECTOR_DB is None:
        return {"present": False}

    data = _VECTOR_DB.get()
    return {
        "present": True,
        "num_chunks": len(data.get("ids", []))
    }
