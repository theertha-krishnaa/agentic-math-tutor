import re

MATH_KEYWORDS = {
    "solve", "calculate", "compute", "evaluate", "simplify", "expand",
    "factor", "differentiate", "integrate", "derive", "prove", "verify",
    "integral", "derivative", "limit", "gradient", "divergence",
    "equation", "inequality", "formula", "theorem", "expression",
    "polynomial", "quadratic", "linear", "cubic", "logarithm", "log",
    "matrix", "vector", "determinant", "eigenvalue", "transpose",
    "probability", "permutation", "combination", "binomial", "variance",
    "algebra", "calculus", "geometry", "trigonometry", "statistics",
    "arithmetic", "sequence", "series", "function", "graph",
    "sin", "cos", "tan", "sine", "cosine", "tangent",
    "hypotenuse", "triangle", "circle", "parabola", "ellipse",
    "area", "volume", "perimeter", "circumference", "radius", "diameter",
    "mean", "median", "mode", "deviation", "distribution",
    "prime", "factorial", "fibonacci", "exponent", "root", "square root",
    "fraction", "decimal", "percentage", "ratio", "proportion",
    "slope", "intercept", "coordinate", "axis", "angle",
    "maximum", "minimum", "optimize", "converge", "diverge",
    "addition", "subtraction", "multiplication", "division",
    "plus", "minus", "multiply", "divided by", "squared", "cubed",
    "inverse", "orthogonal", "perpendicular", "parallel",
    "digit", "integer", "rational", "irrational", "complex number",
    "real number", "natural number", "infinity",
}

BLOCKLIST = {
    "porn", "sex", "nude", "naked", "kill", "murder", "suicide",
    "drugs", "hack", "weapon", "bomb", "terrorist", "racist",
    "password", "credit card", "ssn", "social security",
}

NON_MATH_PATTERNS = [
    r"^what is (the )?(capital|president|population|currency|language|flag)",
    r"^who (is|was|are|were)",
    r"^when (did|was|is|are)",
    r"^where (is|are|was|were)",
    r"^why (is|are|was|were|did|do)",
    r"^(tell me|write|compose|create|give me|list|name|describe)",
    r"^(what|who|where|when|why|how) (is|are|was|were) (the )?(best|worst|most|famous|popular|capital|president|king|queen)",
]


def validate_input(question: str) -> dict:
    if not question or not question.strip():
        return {"allowed": False, "reason": "Question cannot be empty."}

    q = question.lower().strip()

    if len(q) < 5:
        return {"allowed": False, "reason": "Question is too short."}

    if len(q) > 1000:
        return {"allowed": False, "reason": "Question is too long. Please keep it under 1000 characters."}

    for word in BLOCKLIST:
        if word in q:
            return {"allowed": False, "reason": "This content is not allowed."}

    for pattern in NON_MATH_PATTERNS:
        if re.match(pattern, q):
            return {
                "allowed": False,
                "reason": "Only math questions are supported. Please ask about algebra, calculus, geometry, statistics, or similar topics."
            }

    has_math = any(kw in q for kw in MATH_KEYWORDS)
    if not has_math:
        return {
            "allowed": False,
            "reason": "Only math questions are supported. Please ask about algebra, calculus, geometry, statistics, or similar topics."
        }

    return {"allowed": True, "reason": "OK"}


def sanitize_output(response: str) -> str:
    if not response:
        return "I was unable to generate an answer. Please try again."

    text = re.sub(r'<[^>]+>', '', response)

    bad_phrases = [
        "in my opinion", "i believe", "religiously", "culturally",
        "in my culture", "you should", "i think you", "personally",
        "in my experience", "as a human", "as an ai, i feel",
    ]
    lines = text.split('\n')
    cleaned_lines = [
        line for line in lines
        if not any(phrase in line.lower() for phrase in bad_phrases)
    ]
    text = '\n'.join(cleaned_lines)

    if len(text) > 2000:
        text = text[:2000]
        last_end = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
        if last_end > 100:
            text = text[:last_end + 1]

    return text.strip()