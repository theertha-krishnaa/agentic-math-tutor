import os
import logging
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama3-70b-8192")
MAX_SEARCH_RESULTS  = int(os.getenv("MAX_SEARCH_RESULTS", "3"))

MATH_SYSTEM_PROMPT = """You are a precise and helpful math tutor.

Rules:
- Solve math problems step by step, showing all working clearly
- Use clear notation and explain each step
- Always give a final, clearly labelled answer
- If the question is NOT about mathematics, respond with exactly: NOT_MATH
- Never discuss politics, religion, personal opinions, or non-math topics
- Be concise but complete
"""


class GroqLLM:
    """Wrapper around Groq API for Llama3 inference."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        logger.info("GroqLLM ready — model: %s", GROQ_MODEL)

    def solve(self, question: str, context: str = "") -> dict:
        """
        Send a math question to Llama3 via Groq.

        Args:
            question: The math question.
            context:  Optional context from vector DB or web search.

        Returns:
            {"answer": str, "model": str}
        """
        if context:
            user_message = (
                f"Use the following context to help answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            user_message = question

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": MATH_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,      # low = deterministic math answers
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            return {"answer": answer, "model": GROQ_MODEL}

        except Exception as e:
            logger.error("Groq API error: %s", e)
            return {"answer": "", "model": GROQ_MODEL, "error": str(e)}


class TavilySearch:
    """Wrapper around Tavily web search API."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY not set in .env")
        self.client = TavilyClient(api_key=api_key)
        logger.info("TavilySearch ready")

    def search(self, query: str) -> str:
        """
        Search the web for math-related content.

        Returns:
            Combined text from top results, or empty string if search fails.
        """
        try:
            # Add "math" to ensure relevant results
            math_query  = f"math {query}" if "math" not in query.lower() else query
            results     = self.client.search(
                query        = math_query,
                search_depth = "advanced",
                max_results  = MAX_SEARCH_RESULTS,
            )
            # Combine content from all results
            contents = [r.get("content", "") for r in results.get("results", []) if r.get("content")]
            combined = "\n\n".join(contents)
            logger.debug("Tavily returned %d results", len(contents))
            return combined

        except Exception as e:
            logger.warning("Tavily search failed: %s", e)
            return ""   # agent will fall back to LLM-only