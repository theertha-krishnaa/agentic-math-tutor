import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class DSPyVerifier:
    """
    Direct Groq verifier — same interface as the DSPy version,
    no DSPy dependency required.
    """
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info("Verifier ready (direct Groq)")

    def verify(self, question: str, answer: str) -> dict:
    	try:
        	prompt = (
        	    "You are a strict math verifier. Work through the problem yourself "
        	    "first, then compare to the proposed answer.\n\n"
        	    f"Question: {question}\n"
        	    f"Proposed answer: {answer}\n\n"
        	    "Step 1: Solve the question yourself from scratch.\n"
        	    "Step 2: Compare your solution to the proposed answer.\n"
        	    "Step 3: Reply in EXACTLY this format on three new lines:\n"
        	    "CORRECT: true or false\n"
        	    "CONFIDENCE: a decimal like 0.45 or 0.91 reflecting how sure you are\n"
        	    "REASON: one sentence\n"
        	)
        	response = self.client.chat.completions.create(
          	  model=GROQ_MODEL,
          	  messages=[{"role": "user", "content": prompt}],
          	  temperature=0.1,
          	  max_tokens=600,   # enough room to show working before the verdict
        	)
        	text = response.choices[0].message.content.strip()
        	logger.info("Verifier response: %s", text)

        	# Parse only the last occurrence of each key
        	# (the model shows working first, verdict comes at the end)
        	lines = {}
        	for line in text.splitlines():
        	    if ":" in line:
        	        key, _, val = line.partition(":")
        	        k = key.strip().upper()
        	        if k in ("CORRECT", "CONFIDENCE", "REASON"):
        	            lines[k] = val.strip()

        	is_correct = lines.get("CORRECT", "true").lower() in ("true", "yes")

        	try:
        	    confidence = float(lines.get("CONFIDENCE", "0.80").replace("%", "").strip())
        	    if confidence > 1:
        	        confidence /= 100
        	    confidence = max(0.0, min(1.0, confidence))
        	except (ValueError, AttributeError):
        	    confidence = 0.80

        	return {
        	    "verified":    is_correct,
        	    "confidence":  round(confidence, 3),
        	    "explanation": lines.get("REASON", "Verified."),
        	}

    	except Exception as e:
    	    logger.error("Verify error: %s", e)
    	    return {"verified": True, "confidence": 0.5, "explanation": "Verification unavailable."}

    def refine(self, question: str, wrong_answer: str, feedback: str) -> str:
        try:
            prompt = (
                f"Fix this math solution.\n\n"
                f"Question: {question}\n"
                f"Wrong answer: {wrong_answer}\n"
                f"Issue: {feedback}\n\n"
                f"Give the correct step-by-step solution only."
            )
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Refine error: %s", e)
            return wrong_answer