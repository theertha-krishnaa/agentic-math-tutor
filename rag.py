import os
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "math_knowledge")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
VECTOR_SIZE          = 384
EMBED_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"

# Vercel's filesystem is read-only except /tmp — fastembed must cache there
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")

_embedder_instance = None

def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from fastembed import TextEmbedding
        logger.info("Loading fastembed model...")
        _embedder_instance = TextEmbedding(model_name=EMBED_MODEL)
        logger.info("Fastembed model loaded.")
    return _embedder_instance


class QdrantManager:
    def __init__(self):
        url     = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Qdrant connected via cloud URL")
        else:
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
            logger.info("Qdrant connected via host:port")

        self._ensure_collection()
        logger.info("QdrantManager ready — collection: %s", COLLECTION_NAME)

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s", COLLECTION_NAME)

    def _embed(self, text: str) -> list:
        embedder = get_embedder()
        vectors  = list(embedder.embed([text]))
        return vectors[0].tolist()

    def add_document(self, text: str, metadata: dict = None) -> str:
        vector   = self._embed(text)
        point_id = str(uuid.uuid4())
        payload  = {"text": text, **(metadata or {})}
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def add_qa_pair(self, question: str, answer: str, source: str = "llm") -> str:
        vector   = self._embed(question)
        point_id = str(uuid.uuid4())
        payload  = {
            "text":     answer,
            "question": question,
            "source":   source,
            "type":     "qa_pair",
        }
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.info("Stored QA pair — source=%s id=%s", source, point_id[:8])
        return point_id

    def search(self, query: str, top_k: int = 3) -> list:
        vector  = self._embed(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
        )
        return [
            {
                "text":     r.payload.get("text", ""),
                "score":    round(r.score, 4),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in results.points
        ]

    def get_best_match(self, query: str) -> dict | None:
        results = self.search(query, top_k=1)
        if results and results[0]["score"] >= CONFIDENCE_THRESHOLD:
            return results[0]
        return None

    def count(self) -> int:
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count