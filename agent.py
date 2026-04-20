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
          2. If found (score >= threshold) → verify before trusting
          3. If verified → return cached answer immediately
          4. If not found or verification failed → call Groq LLM
          5. Store the new LLM answer in Qdrant
          6. Verify and optionally refine
          7. Return structured result
        """
        refinement_applied = False

        # ── Step 1: Search vector DB ──────────────────────────────────────────
        logger.info("Searching Qdrant for: %s", question[:60])
        db_result = self.rag.get_best_match(question)

        if db_result:
            logger.info("DB hit — score: %.3f", db_result["score"])

            # ── Step 2: Verify the cached answer before trusting it ───────────
            pre_verification = self.verifier.verify(question, db_result["text"])

            if pre_verification["is_correct"] and pre_verification["confidence"] >= REFINEMENT_THRESHOLD:
                # ── Step 3: Cache hit is valid — return immediately ───────────
                logger.info(
                    "DB hit verified — confidence=%.3f, returning cached answer",
                    pre_verification["confidence"],
                )
                return {
                    "answer":             db_result["text"],
                    "source":             "vector_db",
                    "verified":           True,
                    "confidence":         pre_verification["confidence"],
                    "explanation":        pre_verification["explanation"],
                    "refinement_applied": False,
                    "stored_in_db":       False,
                }

            # Verifier rejected the cached answer — fall through to LLM
            logger.info(
                "DB hit rejected by verifier (confidence=%.3f, is_correct=%s) — falling back to LLM",
                pre_verification["confidence"],
                pre_verification["is_correct"],
            )

        # ── Step 4: LLM fallback ──────────────────────────────────────────────
        logger.info("DB miss or cache rejected — calling Groq LLM...")
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

        # ── Step 5: Store in Qdrant so next time it's a DB hit ───────────────
        logger.info("Storing new answer in Qdrant...")
        self.rag.add_qa_pair(
            question = question,
            answer   = answer,
            source   = "llm",
        )
        source = "llm_generated"

        # ── Step 6: Verify ────────────────────────────────────────────────────
        logger.info("Verifying answer...")
        verification = self.verifier.verify(question, answer)

        # ── Step 7: Refine if confidence is low ───────────────────────────────
        if verification["confidence"] < REFINEMENT_THRESHOLD:
            logger.info("Low confidence (%.3f) — refining...", verification["confidence"])
            refined = self.verifier.refine(
                question     = question,
                wrong_answer = answer,
                feedback     = verification["explanation"],
            )
            if refined and refined != answer:
                self.rag.add_qa_pair(
                    question = question,
                    answer   = refined,
                    source   = "llm_refined",
                )
                answer             = refined
                refinement_applied = True
                source             = "llm_refined"

        return {
            "answer":             answer,
            "source":             source,
            "verified":           verification["verified"],
            "confidence":         verification["confidence"],
            "explanation":        verification["explanation"],
            "refinement_applied": refinement_applied,
            "stored_in_db":       True,
        }
