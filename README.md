# Agentic Math Tutor — Agent-RAG System

> A production-grade agentic RAG backend that answers math questions using a Qdrant vector knowledge base, Groq Llama3 LLM fallback, and DSPy verification — with strict input/output guardrails.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat)
![Groq](https://img.shields.io/badge/Groq-Llama3_70B-F55036?style=flat)
![DSPy](https://img.shields.io/badge/DSPy-2.4-7C3AED?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What It Does

A student sends a math question → the system searches a vector knowledge base for a similar answered question → if found, returns it instantly → if not, calls Llama3, stores the answer back in the database, and returns it. Every answer is verified by a DSPy pipeline before being shown.

**The core loop is true RAG: retrieve → generate only if needed → store back.**

---

## Architecture
```
Question
    │
    ▼
[Input Guardrails]
Blocks non-math queries,
offensive content, prompt injection
    │
    ▼
[Qdrant Vector Search]
Semantic similarity search
over stored QA pairs
    │
score ≥ 0.75?
    │
   YES ──────────────────── return cached answer
    │
   NO
    │
    ▼
[Groq Llama3-70B]
Generate step-by-step solution
temperature = 0.1 (deterministic)
    │
    ▼
[Write back to Qdrant]
Store question + answer as vector
so next identical/similar query
hits the DB instead of LLM
    │
    ▼
[DSPy Verifier]
Structured verification pipeline
confidence score 0.0 → 1.0
auto-refine if confidence < 0.6
    │
    ▼
[Output Guardrails]
Sanitize response, strip HTML,
remove non-math content, cap length
    │
    ▼
JSON response to client
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM Inference | Groq API — Llama3-70B | Fast, cheap inference at 700+ tokens/sec |
| Vector Database | Qdrant (Docker) | Semantic similarity search over QA pairs |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` | 384-dim vectors for question matching |
| LLM Pipeline | DSPy 2.4 | Structured verification and answer refinement |
| Backend | FastAPI + Pydantic | REST API with typed request/response models |
| Guardrails | Custom NLP filters | Input validation + output sanitization |
| Frontend | Vanilla HTML/CSS/JS | Zero-dependency test UI |

---

## Project Structure
```
math-tutor/
├── main.py               # FastAPI app — endpoints, startup, middleware
├── agent.py              # MCPRouter — RAG pipeline orchestrator
├── rag.py                # QdrantManager — vector DB read/write
├── tools.py              # GroqLLM — Llama3 inference wrapper
├── dspy_verifier.py      # DSPyVerifier — answer verification + refinement
├── guardrails.py         # Input validation + output sanitization
├── seed_knowledge.py     # Populates Qdrant with base math knowledge
├── frontend/
│   └── index.html        # Test UI (no framework, no build step)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/agentic-math-tutor.git
cd agentic-math-tutor
```

### 2. Create environment
```bash
conda create -n math-tutor python=3.10 -y
conda activate math-tutor
pip install -r requirements.txt
```

### 3. Get API keys

| Service | URL | Free tier |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day |
| Tavily | [app.tavily.com](https://app.tavily.com) | 1,000 searches/month |

### 4. Configure environment
```bash
cp .env.example .env
```

Edit `.env` with your keys:
```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=math_knowledge
CONFIDENCE_THRESHOLD=0.75
```

### 5. Start Qdrant
```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 6. Seed the knowledge base
```bash
python seed_knowledge.py
```

Loads 18 math documents covering algebra, calculus, geometry, statistics, trigonometry, linear algebra, probability, and sequences.

### 7. Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Open the frontend

Open `frontend/index.html` directly in Chrome or Edge.  
Or use the auto-generated API docs at `http://localhost:8000/docs`.

---

## API Endpoints

### `POST /ask`

Submit a math question. Returns a verified answer with source and confidence.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Solve x² + 5x + 6 = 0"}'
```
```json
{
  "answer": "Using the quadratic formula...\nx = -2 or x = -3",
  "source": "llm_generated",
  "verified": true,
  "confidence": 0.92,
  "explanation": "Both roots satisfy the original equation.",
  "refinement_applied": false,
  "stored_in_db": true
}
```

**Source values:**

| Value | Meaning |
|---|---|
| `vector_db` | Retrieved from Qdrant — no LLM call made |
| `llm_generated` | Generated by Llama3, now stored in DB |
| `llm_refined` | Generated + refined by DSPy due to low confidence |

### `POST /add-knowledge`

Add a document to the knowledge base at runtime.
```bash
curl -X POST http://localhost:8000/add-knowledge \
  -H "Content-Type: application/json" \
  -d '{"text": "The fundamental theorem of calculus...", "topic": "calculus", "difficulty": "hard"}'
```

### `GET /health`
### `GET /stats`

---

## How the RAG Loop Works

**First time asking "Solve x² + 5x + 6 = 0":**
1. Qdrant search finds no match above 0.75 threshold
2. Groq Llama3 solves it (≈1–2 seconds)
3. Answer stored in Qdrant as a QA vector pair
4. DSPy verifies — returns confidence 0.92
5. Response: `source: llm_generated`, `stored_in_db: true`

**Second time asking the same (or similar) question:**
1. Qdrant search finds the stored answer at score 0.94
2. No LLM call made — instant retrieval
3. Response: `source: vector_db`, `stored_in_db: false`

This is the write-back cache pattern — the system gets faster and cheaper over time as the DB fills up.

---

## Guardrails

**Input — blocked queries:**
- Non-math questions (`"What is the capital of France?"` → blocked)
- Questions matching non-math patterns (`"Who is..."`, `"Tell me about..."`)
- Offensive or harmful content
- Prompt injection attempts (>1000 characters)

**Output — sanitization:**
- HTML tag stripping
- Personal opinion / cultural bias removal
- Response capped at 2000 characters on sentence boundary

---

## DSPy Verification

DSPy is used for structured LLM verification rather than free-form prompting. Two typed signatures are defined:

`MathVerifier` — takes question + answer, returns `is_correct`, `confidence` (0–1), `explanation`

`MathRefiner` — takes question + wrong answer + feedback, returns `refined_answer` + `steps`

If DSPy configuration fails for the installed version, the system automatically falls back to a direct structured Groq call with the same schema — so verification always runs regardless of DSPy version compatibility.

---

## License

MIT — free to use, modify, and distribute.

---

## Author

Built by [Theertha Krishna](https://github.com/theertha-krishnaa)  
Connect on [LinkedIn](https://www.linkedin.com/in/theertha-krishna-5003b1256/)