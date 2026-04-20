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

# ── Allowed origins ───────────────────────────────────────────────────────────
# Add your Framer portfolio URL here once deployed, e.g.:
# "https://yourname.framer.app"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading agent on startup...")
    from rag import get_encoder
    get_encoder()                    # ← warm up the model now
    app.state.router = MCPRouter()
    logger.info("Agent ready ✓")
    yield
    logger.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Agentic Math Tutor",
    description = "Agent-RAG system using Qdrant Cloud, Groq Llama3, and DSPy",
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
@app.get("/")
def root():
    return {"message": "Agentic Math Tutor API is running. POST /ask to use it."}

@app.get("/health")
def health():
    """Check if the server and Qdrant are running."""
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
    """Submit a math question, get a verified answer."""
    check = validate_input(body.question)
    if not check["allowed"]:
        raise HTTPException(status_code=400, detail=check["reason"])

    result = app.state.router.route(body.question)
    result["answer"] = sanitize_output(result["answer"])
    return result

@app.post("/add-knowledge")
async def add_knowledge(body: KnowledgeRequest):
    """Add a new math document to the Qdrant knowledge base."""
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
    """Return knowledge base statistics."""
    try:
        rag = QdrantManager()
        return {
            "documents_in_db":      rag.count(),
            "model":                os.getenv("GROQ_MODEL"),
            "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
            "vector_model":         "all-MiniLM-L6-v2",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-verify")
async def debug_verify():
    from dspy_verifier import DSPyVerifier
    import traceback
    try:
        v = DSPyVerifier()
        result = v.verify("What is 2 + 2?", "The answer is 4.")
        return {"result": result, "fallback": getattr(v, '_use_fallback', False)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/debug-verify2")
async def debug_verify2():
    import traceback
    import dspy
    from dspy_verifier import DSPyVerifier, MathVerifier
    try:
        v = DSPyVerifier()
        predictor = dspy.Predict(MathVerifier)
        result = predictor(question="What is 2+2?", answer="4")
        return {"success": True, "is_correct": result.is_correct, "confidence": result.confidence, "explanation": result.explanation}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
