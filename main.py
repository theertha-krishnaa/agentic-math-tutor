import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from guardrails import validate_input, sanitize_output
from agent import MCPRouter
from rag import QdrantManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting...")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title       = "Agentic Math Tutor",
    description = "Agent-RAG system using Qdrant Cloud, Groq Llama3",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

def get_router():
    if not hasattr(get_router, '_instance'):
        logger.info("Initialising MCPRouter...")
        get_router._instance = MCPRouter()
        logger.info("MCPRouter ready.")
    return get_router._instance

class QuestionRequest(BaseModel):
    question: str

class KnowledgeRequest(BaseModel):
    text:       str
    topic:      str = "general"
    difficulty: str = "medium"

class AnswerResponse(BaseModel):
    answer:             str
    source:             str
    verified:           bool
    confidence:         float
    explanation:        str
    refinement_applied: bool

@app.get("/")
def root():
    return {"message": "Agentic Math Tutor API is running. POST /ask to use it."}

@app.get("/health")
def health():
    try:
        rag   = QdrantManager()
        count = rag.count()
        return {
            "status":    "ok",
            "model":     os.getenv("GROQ_MODEL"),
            "qdrant":    "connected",
            "documents": count,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/ask", response_model=AnswerResponse)
async def ask(body: QuestionRequest):
    check = validate_input(body.question)
    if not check["allowed"]:
        raise HTTPException(status_code=400, detail=check["reason"])
    result = get_router().route(body.question)
    result["answer"] = sanitize_output(result["answer"])
    return result

@app.post("/add-knowledge")
async def add_knowledge(body: KnowledgeRequest):
    try:
        rag    = QdrantManager()
        doc_id = rag.add_document(
            text     = body.text,
            metadata = {"topic": body.topic, "difficulty": body.difficulty},
        )
        return {"added": True, "id": doc_id, "topic": body.topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    try:
        rag = QdrantManager()
        return {
            "documents_in_db":      rag.count(),
            "model":                os.getenv("GROQ_MODEL"),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.85")),
            "vector_model":         "all-MiniLM-L6-v2",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
