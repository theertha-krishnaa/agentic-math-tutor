import os
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "math_knowledge")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
VECTOR_SIZE          = 384


class QdrantManager:

    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self.client  = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
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

    def add_document(self, text: str, metadata: dict = None) -> str:
        """Store any text document."""
        vector   = self.encoder.encode(text).tolist()
        point_id = str(uuid.uuid4())
        payload  = {"text": text, **(metadata or {})}
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.debug("Added document id=%s", point_id)
        return point_id

    def add_qa_pair(self, question: str, answer: str, source: str = "llm") -> str:
        """
        Store a question+answer pair.
        The vector is built from the question so future similar questions
        match against it. The full answer is stored in the payload.
        source = 'llm' means it was generated, 'seeded' means it came from seed_knowledge.py
        """
        vector   = self.encoder.encode(question).tolist()
        point_id = str(uuid.uuid4())
        payload  = {
            "text":     answer,       # what gets returned as the answer
            "question": question,     # stored for reference
            "source":   source,
            "type":     "qa_pair",
        }
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.info("Stored QA pair in Qdrant — source=%s id=%s", source, point_id[:8])
        return point_id

    def search(self, query: str, top_k: int = 3) -> list:
        vector  = self.encoder.encode(query).tolist()
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
        )
        return [
            {
                "text":     r.payload.get("text", ""),
                "score":    round(r.score, 4),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in results
        ]

    def get_best_match(self, query: str) -> dict | None:
        """Return top match only if score meets confidence threshold."""
        results = self.search(query, top_k=1)
        if results and results[0]["score"] >= CONFIDENCE_THRESHOLD:
            return results[0]
        return None

    def count(self) -> int:
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count