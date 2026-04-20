import re

# ── Math keywords — if none of these appear, block the question ──────────────
MATH_KEYWORDS = {
    "solve", "calculate", "find", "prove", "integral", "derivative",
    "equation", "matrix", "probability", "geometry", "algebra",
    "calculus", "statistics", "formula", "theorem", "function",
    "graph", "polynomial", "factor", "simplify", "expand", "evaluate",
    "compute", "limit", "sum", "product", "vector", "angle", "triangle",
    "circle", "area", "volume", "number", "digit", "prime", "root",
    "square", "cube", "power", "exponent", "logarithm", "log", "sin",
    "cos", "tan", "differentiate", "integrate", "series", "sequence",
    "arithmetic", "geometric", "mean", "median", "mode", "variance",
    "standard deviation", "permutation", "combination", "binomial",
    "fraction", "decimal", "percentage", "ratio", "proportion",
    "linear", "quadratic", "cubic", "inequality", "coordinate",
    "slope", "intercept", "parabola", "hyperbola", "ellipse",
    "determinant", "eigenvalue", "eigenvector", "transpose", "inverse",
    "gradient", "divergence", "curl", "laplacian", "fourier",
    "what is", "how to", "explain", "define", "show", "verify",
    "distance", "speed", "rate", "time", "work", "profit", "loss",
    "interest", "tax", "discount", "average", "maximum", "minimum",
    "plus", "minus", "multiply", "divide", "times", "divided",
    "greater", "less", "equal", "x", "y", "z", "n", "k",
}

# ── Blocklist — offensive or off-topic trigger words ─────────────────────────
BLOCKLIST = {
    "porn", "sex", "nude", "naked", "kill", "murder", "suicide",
    "drugs", "hack", "weapon", "bomb", "terrorist", "racist",
    "password", "credit card", "ssn", "social security",
}


def validate_input(question: str) -> dict:
    """
    Check if a question is a valid math question.

    Returns:
        {"allowed": True, "reason": "OK"}
        {"allowed": False, "reason": "explanation"}
    """
    if not question or not question.strip():
        return {"allowed": False, "reason": "Question cannot be empty."}

    q = question.lower().strip()

    # Too short
    if len(q) < 5:
        return {"allowed": False, "reason": "Question is too short."}

    # Too long (prevent prompt injection)
    if len(q) > 1000:
        return {"allowed": False, "reason": "Question is too long. Please keep it under 1000 characters."}

    # Blocklist check
    for word in BLOCKLIST:
        if word in q:
            return {"allowed": False, "reason": "This content is not allowed."}

    # Math keyword check
    has_math = any(kw in q for kw in MATH_KEYWORDS)
    if not has_math:
        return {
            "allowed": False,
            "reason": "Only math questions are supported. Please ask about algebra, calculus, geometry, statistics, or similar topics."
        }

    return {"allowed": True, "reason": "OK"}


def sanitize_output(response: str) -> str:
    """
    Clean and sanitize the LLM response before sending to user.
    """
    if not response:
        return "I was unable to generate an answer. Please try again."

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', response)

    # Remove lines with non-math personal/cultural opinions
    bad_phrases = [
        "in my opinion", "i believe", "religiously", "culturally",
        "in my culture", "you should", "i think you", "personally",
        "in my experience", "as a human", "as an ai, i feel",
    ]
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower()
        if not any(phrase in line_lower for phrase in bad_phrases):
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Cap at 2000 characters, ending on a complete sentence
    if len(text) > 2000:
        text = text[:2000]
        # Find last complete sentence
        last_end = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
        if last_end > 100:
            text = text[:last_end + 1]

    return text.strip()