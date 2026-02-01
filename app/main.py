from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, time, logging
from dotenv import load_dotenv
from threading import Lock
import base64
from typing import Dict
from langchain.schema import Document
from langchain_community.callbacks.manager import get_openai_callback
from app.db import create_vectorstore_from_data, load_vectorstore

# PDF handling
import fitz  # PyMuPDF

# Load env
load_dotenv()

app = FastAPI()
db_lock = Lock()

# ---------------- LOGGING ---------------- #

def setup_logging():
    os.makedirs("logs", exist_ok=True)

    app_logger = logging.getLogger("rag_app")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(logging.StreamHandler())

    query_logger = logging.getLogger("query_logger")
    query_logger.setLevel(logging.INFO)
    query_logger.addHandler(logging.FileHandler("logs/query_logs.log"))
    query_logger.addHandler(logging.StreamHandler())

    training_logger = logging.getLogger("training_logger")
    training_logger.setLevel(logging.INFO)
    training_logger.addHandler(logging.FileHandler("logs/training_logs.log"))
    training_logger.addHandler(logging.StreamHandler())

    return app_logger, query_logger, training_logger

app_logger, query_logger, training_logger = setup_logging()

# ---------------- MODELS ---------------- #

class Query(BaseModel):
    question: str

class TrainRequest(BaseModel):
    url: str | None = None
    files: Dict[str, str] | None = None  # Base64 content

# ---------------- UTILITIES ---------------- #

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += "\n" + page.get_text("text")
    return text

def clean_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())

# ---------------- TRAIN ---------------- #

@app.post("/train")
def train_knowledge_base(payload: TrainRequest):
    from langchain_community.document_loaders import WebBaseLoader

    start_time = time.time()
    training_logger.info(f"Training started | URL={payload.url}")

    docs = []

    if payload.url:
        loader = WebBaseLoader(payload.url)
        docs.extend(loader.load())

    if payload.files:
        if len(payload.files) > 1:
            return JSONResponse(status_code=400, content={"error": "Only one file allowed"})

        for name, b64_content in payload.files.items():
            raw_bytes = base64.b64decode(b64_content)
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw_bytes)
            else:
                text = raw_bytes.decode("utf-8", errors="ignore")

            text = clean_text(text)
            if not text:
                return {"error": f"{name} has no readable content"}

            docs.append(Document(page_content=text, metadata={"source": name}))

    if not docs:
        return {"error": "No data provided for training"}

    with db_lock:
        with get_openai_callback():
            chunks_info = create_vectorstore_from_data(docs)

    return {
        "message": "Knowledge base trained",
        "chunks_created": chunks_info["chunks_created"],
        "estimated_tokens": chunks_info["estimated_embedding_tokens"],
        "estimated_cost_usd": chunks_info["estimated_embedding_cost_usd"],
        "training_time_sec": round(time.time() - start_time, 2)
    }

# ---------------- QUERY ---------------- #

@app.post("/query")
def query_rag(payload: Query):
    start_time = time.time()
    query_logger.info(f"Query: {payload.question}")

    try:
        from app.rag_chain import get_rag_chain

        comparison_hint = (
            "If the question asks for differences or comparisons, "
            "first summarize the main subject, then explain how it differs "
            "from other projects mentioned in the context. "
            "Only use the provided context."
        )

        question_with_hint = f"{comparison_hint}\n\nQuestion: {payload.question}"

        with get_openai_callback() as cb:
            result = get_rag_chain(question_with_hint)

            answer = result if isinstance(result, str) else str(result)

            with db_lock:
                vectordb = load_vectorstore()
                if vectordb:
                    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
                    docs = retriever.invoke(payload.question)

                    # 🔥 FILTER NOISE CHUNKS
                    docs = [
                        d for d in docs
                        if len(d.page_content.strip()) > 120
                        and "Previous" not in d.page_content
                        and "Next" not in d.page_content
                        and "Portfolio" not in d.page_content
                    ]

                    retrieved_chunks = [
                        d.page_content[:200] + "..." if len(d.page_content) > 200 else d.page_content
                        for d in docs
                    ]
                    context_chars = sum(len(d.page_content) for d in docs)
                else:
                    retrieved_chunks = []
                    context_chars = 0

            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {
                    "latency_sec": round(time.time() - start_time, 2),
                    "tokens_used": cb.total_tokens,
                    "cost_usd": cb.total_cost,
                    "chunks_retrieved": len(retrieved_chunks),
                    "context_chars_used": context_chars
                }
            }

    except Exception as e:
        query_logger.error(str(e))
        return {"error": f"Query failed: {str(e)}"}

# ---------------- HEALTH ---------------- #

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    app_logger.info("RAG API started")

@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("RAG API stopped")
