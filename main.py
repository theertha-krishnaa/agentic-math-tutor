import os
import logging
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

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Agentic Math Tutor",
    description = "Agent-RAG system using Qdrant, Tavily, Groq/Llama3, and DSPy",
    version     = "1.0.0",
)

# Allow the frontend HTML file to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load agent once at startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Loading agent...")
    app.state.router = MCPRouter()
    logger.info("Agent ready")


# ── Request / Response models ─────────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        count = app.state.router.rag.count()
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
    """
    Main endpoint. Submit a math question, get a verified answer.
    Non-math questions are blocked by input guardrails.
    """
    # 1. Input guardrail
    check = validate_input(body.question)
    if not check["allowed"]:
        raise HTTPException(status_code=400, detail=check["reason"])

    # 2. Route through agent
    result = app.state.router.route(body.question)

    # 3. Output guardrail
    result["answer"] = sanitize_output(result["answer"])

    return result


@app.post("/add-knowledge")
async def add_knowledge(body: KnowledgeRequest):
    """Add a new math document to the Qdrant knowledge base."""
    try:
        rag = QdrantManager()
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
        router = app.state.router
        return {
            "documents_in_db":      router.rag.count(),
            "model":                os.getenv("GROQ_MODEL"),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
            "vector_model":         "all-MiniLM-L6-v2",
        }
    except AttributeError:
        raise HTTPException(status_code=503, detail="Agent not initialised yet. Try again in a moment.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))