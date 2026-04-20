import logging
from rag import QdrantManager
from tools import GroqLLM
from dspy_verifier import DSPyVerifier

logger = logging.getLogger(__name__)

REFINEMENT_THRESHOLD = 0.6


class MCPRouter:

    def __init__(self):
        logger.info("Initialising MCPRouter...")
        self.rag      = QdrantManager()
        self.llm      = GroqLLM()
        self.verifier = DSPyVerifier()
        logger.info("MCPRouter ready")

    def route(self, question: str) -> dict:
        """
        True RAG loop:
          1. Search Qdrant for a similar answered question
          2. If found (score >= threshold) → return it directly
          3. If not found → call Groq LLM
          4. Store the new LLM answer in Qdrant
          5. Verify and optionally refine
          6. Return structured result
        """
        refinement_applied = False

        # ── Step 1: Search vector DB ──────────────────────────────────────────
        logger.info("Searching Qdrant for: %s", question[:60])
        db_result = self.rag.get_best_match(question)

        if db_result:
            logger.info("DB hit — score: %.3f", db_result["score"])
            answer = db_result["text"]
            source = "vector_db"

        else:
            # ── Step 2: LLM fallback ──────────────────────────────────────────
            logger.info("DB miss — calling Groq LLM...")
            llm_result = self.llm.solve(question)
            print("LLM RESULT:", llm_result) 
            answer     = llm_result.get("answer", "")

            if not answer:
                return {
                    "answer":             "I was unable to generate an answer. Please try again.",
                    "source":             "llm_only",
                    "verified":           False,
                    "confidence":         0.0,
                    "explanation":        "LLM returned empty response.",
                    "refinement_applied": False,
                    "stored_in_db":       False,
                }

            # ── Step 3: Store in Qdrant so next time it's a DB hit ────────────
            logger.info("Storing new answer in Qdrant...")
            self.rag.add_qa_pair(
                question = question,
                answer   = answer,
                source   = "llm",
            )
            source = "llm_generated"

        # ── Step 4: Verify ────────────────────────────────────────────────────
        logger.info("Verifying answer...")
        verification = self.verifier.verify(question, answer)

        # ── Step 5: Refine if confidence is low ───────────────────────────────
        if verification["confidence"] < REFINEMENT_THRESHOLD:
            logger.info("Low confidence (%.3f) — refining...", verification["confidence"])
            refined = self.verifier.refine(
                question     = question,
                wrong_answer = answer,
                feedback     = verification["explanation"],
            )
            if refined and refined != answer:
                # Store the refined answer too, replacing intent
                self.rag.add_qa_pair(question=question, answer=refined, source="llm_refined")
                answer             = refined
                refinement_applied = True

        return {
            "answer":             answer,
            "source":             source,
            "verified":           verification["verified"],
            "confidence":         verification["confidence"],
            "explanation":        verification["explanation"],
            "refinement_applied": refinement_applied,
            "stored_in_db":       source != "vector_db",
        }
