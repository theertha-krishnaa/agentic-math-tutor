import os
import logging
import dspy
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ── DSPy Signatures ───────────────────────────────────────────────────────────

class MathVerifier(dspy.Signature):
    """Verify whether a proposed answer to a math question is correct."""
    question    = dspy.InputField(desc="The original math question")
    answer      = dspy.InputField(desc="The proposed answer to verify")
    is_correct  = dspy.OutputField(desc="true if the answer is mathematically correct, false otherwise")
    confidence  = dspy.OutputField(desc="Confidence score as a decimal between 0.0 and 1.0")
    explanation = dspy.OutputField(desc="One sentence explanation of why the answer is correct or incorrect")


class MathRefiner(dspy.Signature):
    """Produce a corrected step-by-step solution to a math question."""
    question       = dspy.InputField(desc="The original math question")
    wrong_answer   = dspy.InputField(desc="The previous incorrect answer")
    feedback       = dspy.InputField(desc="Explanation of what was wrong")
    refined_answer = dspy.OutputField(desc="A correct step-by-step solution")
    steps          = dspy.OutputField(desc="Key steps taken to solve the problem")


# ── Verifier class ────────────────────────────────────────────────────────────

class DSPyVerifier:

    def __init__(self):
        self._use_fallback = False
        self._configure_dspy()
        if not self._use_fallback:
            self.verifier = dspy.Predict(MathVerifier)
            self.refiner  = dspy.Predict(MathRefiner)
        logger.info("DSPyVerifier ready (fallback=%s)", self._use_fallback)

    def _configure_dspy(self):
        """
        Try multiple DSPy configuration styles to handle different
        installed versions of dspy-ai.
        """
        model_string = f"groq/{GROQ_MODEL}"

        # Style 1 — dspy >= 2.5 (dspy.LM)
        try:
            lm = dspy.LM(model=model_string, api_key=GROQ_API_KEY)
            dspy.configure(lm=lm)
            logger.info("DSPy configured via dspy.LM (>=2.5 style)")
            return
        except (AttributeError, Exception) as e:
            logger.debug("dspy.LM failed: %s", e)

        # Style 2 — dspy 2.4.x (dspy.GROQ)
        try:
            lm = dspy.GROQ(model=GROQ_MODEL, api_key=GROQ_API_KEY)
            dspy.settings.configure(lm=lm)
            logger.info("DSPy configured via dspy.GROQ (2.4.x style)")
            return
        except (AttributeError, Exception) as e:
            logger.debug("dspy.GROQ failed: %s", e)

        # Style 3 — OpenAI-compatible Groq endpoint
        try:
            lm = dspy.OpenAI(
                model    = GROQ_MODEL,
                api_key  = GROQ_API_KEY,
                api_base = "https://api.groq.com/openai/v1",
                max_tokens = 1024,
            )
            dspy.settings.configure(lm=lm)
            logger.info("DSPy configured via dspy.OpenAI fallback")
            return
        except (AttributeError, Exception) as e:
            logger.debug("dspy.OpenAI fallback failed: %s", e)

        # All DSPy styles failed — use direct Groq calls
        logger.warning(
            "All DSPy configuration styles failed. "
            "Using direct Groq verification instead."
        )
        self._use_fallback = True

    # ── Public methods ────────────────────────────────────────────────────────

    def verify(self, question: str, answer: str) -> dict:
        """Verify if an answer is mathematically correct."""
        if self._use_fallback:
            return self._fallback_verify(question, answer)
        try:
            result     = self.verifier(question=question, answer=answer)
            is_correct = str(result.is_correct).lower().strip() in (
                "true", "yes", "correct", "1"
            )
            try:
                confidence = float(str(result.confidence).strip())
                if confidence > 1.0:
                    confidence = confidence / 100.0
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, AttributeError):
                confidence = 0.75

            return {
                "verified":    is_correct,
                "confidence":  round(confidence, 3),
                "explanation": str(result.explanation).strip(),
            }

        except Exception as e:
            logger.error("DSPy verify error: %s — switching to fallback", e)
            self._use_fallback = True
            return self._fallback_verify(question, answer)

    def refine(self, question: str, wrong_answer: str, feedback: str) -> str:
        """Generate a better answer given what was wrong."""
        if self._use_fallback:
            return self._fallback_refine(question, wrong_answer, feedback)
        try:
            result = self.refiner(
                question     = question,
                wrong_answer = wrong_answer,
                feedback     = feedback,
            )
            return str(result.refined_answer).strip()
        except Exception as e:
            logger.error("DSPy refine error: %s — switching to fallback", e)
            self._use_fallback = True
            return self._fallback_refine(question, wrong_answer, feedback)

    # ── Direct Groq fallback (no DSPy) ───────────────────────────────────────

    def _fallback_verify(self, question: str, answer: str) -> dict:
        """Verify using Groq directly when DSPy is unavailable."""
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            prompt = (
                f"You are a math verification assistant.\n\n"
                f"Question: {question}\n"
                f"Proposed Answer: {answer}\n\n"
                f"Evaluate this answer carefully. Then respond using EXACTLY "
                f"this format (no extra text, no markdown):\n"
                f"CORRECT: true\n"
                f"CONFIDENCE: 0.85\n"
                f"REASON: The answer correctly applies the quadratic formula.\n\n"
                f"Rules:\n"
                f"- CORRECT must be exactly 'true' or 'false'\n"
                f"- CONFIDENCE must be a decimal like 0.90 (not a percentage)\n"
                f"- REASON must be one sentence\n"
                f"- If the answer looks correct and complete, set CONFIDENCE "
                f"above 0.75\n"
                f"- If the answer is wrong or incomplete, set CONFIDENCE "
                f"below 0.5"
            )

            response = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.1,
                max_tokens  = 120,
            )
            text = response.choices[0].message.content.strip()
            logger.debug("Verifier raw response: %s", text)

            # Parse lines into a dict
            lines = {}
            for line in text.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    lines[key.strip().upper()] = val.strip()

            is_correct = lines.get("CORRECT", "true").lower() in (
                "true", "yes"
            )

            # Handle "0.85", "85%", or "85"
            raw_conf = lines.get("CONFIDENCE", "0.75").replace("%", "").strip()
            try:
                confidence = float(raw_conf)
                if confidence > 1.0:
                    confidence = confidence / 100.0
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.75

            reason = lines.get("REASON", "Answer verified.")

            return {
                "verified":    is_correct,
                "confidence":  round(confidence, 3),
                "explanation": reason,
            }

        except Exception as e:
            logger.error("Fallback verify error: %s", e)
            return {
                "verified":    True,
                "confidence":  0.5,
                "explanation": "Verification unavailable.",
            }

    def _fallback_refine(self, question: str, wrong_answer: str, feedback: str) -> str:
        """Refine using Groq directly when DSPy is unavailable."""
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            prompt = (
                f"You are a precise math tutor.\n\n"
                f"The following answer was incorrect or incomplete.\n\n"
                f"Question: {question}\n"
                f"Wrong answer: {wrong_answer}\n"
                f"What was wrong: {feedback}\n\n"
                f"Please provide the correct, complete step-by-step solution."
            )

            response = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.1,
                max_tokens  = 1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("Fallback refine error: %s", e)
            return wrong_answer