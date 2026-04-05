"""
Run this once before starting the server to populate Qdrant
with a base set of math knowledge.

Usage:
    python seed_knowledge.py
"""

from rag import QdrantManager

KNOWLEDGE_BASE = [
    {
        "text": "Quadratic formula: For ax²+bx+c=0, x = (-b ± √(b²-4ac)) / 2a. The discriminant b²-4ac determines roots: positive = two real roots, zero = one repeated root, negative = two complex roots.",
        "topic": "algebra", "difficulty": "medium"
    },
    {
        "text": "Pythagorean theorem: In a right triangle, a² + b² = c² where c is the hypotenuse. To find hypotenuse: c = √(a²+b²). To find a leg: a = √(c²-b²).",
        "topic": "geometry", "difficulty": "easy"
    },
    {
        "text": "Derivative rules: Power rule d/dx(xⁿ) = nxⁿ⁻¹. Constant rule d/dx(c) = 0. Sum rule d/dx(f+g) = f'+g'. Product rule d/dx(fg) = f'g+fg'. Quotient rule d/dx(f/g) = (f'g-fg')/g². Chain rule d/dx(f(g(x))) = f'(g(x))·g'(x).",
        "topic": "calculus", "difficulty": "hard"
    },
    {
        "text": "Integration rules: ∫xⁿdx = xⁿ⁺¹/(n+1)+C (n≠-1). ∫eˣdx = eˣ+C. ∫(1/x)dx = ln|x|+C. ∫sin(x)dx = -cos(x)+C. ∫cos(x)dx = sin(x)+C.",
        "topic": "calculus", "difficulty": "hard"
    },
    {
        "text": "Mean, median, mode: Mean = sum of values / count. Median = middle value when sorted (average of two middle values if even count). Mode = most frequently occurring value.",
        "topic": "statistics", "difficulty": "easy"
    },
    {
        "text": "Standard deviation: σ = √(Σ(xᵢ-μ)²/N) for population. s = √(Σ(xᵢ-x̄)²/(n-1)) for sample. Variance is σ² or s². Measures spread of data around mean.",
        "topic": "statistics", "difficulty": "medium"
    },
    {
        "text": "Probability rules: P(A) is between 0 and 1. P(A') = 1-P(A). P(A∪B) = P(A)+P(B)-P(A∩B). P(A∩B) = P(A)·P(B) if A and B are independent. Conditional probability P(A|B) = P(A∩B)/P(B).",
        "topic": "probability", "difficulty": "medium"
    },
    {
        "text": "Permutations and combinations: nPr = n!/(n-r)! for ordered arrangements. nCr = n!/(r!(n-r)!) for unordered selections. 0! = 1 by definition.",
        "topic": "combinatorics", "difficulty": "medium"
    },
    {
        "text": "Trigonometry: sin(θ)=opposite/hypotenuse, cos(θ)=adjacent/hypotenuse, tan(θ)=opposite/adjacent. Identities: sin²θ+cos²θ=1, tan θ=sin θ/cos θ. Special angles: sin(30°)=0.5, sin(45°)=√2/2, sin(60°)=√3/2.",
        "topic": "trigonometry", "difficulty": "medium"
    },
    {
        "text": "Arithmetic sequences: aₙ = a₁ + (n-1)d where d is common difference. Sum of n terms: Sₙ = n/2(a₁+aₙ) or Sₙ = n/2(2a₁+(n-1)d).",
        "topic": "sequences", "difficulty": "medium"
    },
    {
        "text": "Geometric sequences: aₙ = a₁·rⁿ⁻¹ where r is common ratio. Sum of n terms: Sₙ = a₁(1-rⁿ)/(1-r) for r≠1. Infinite sum: S∞ = a₁/(1-r) only if |r|<1.",
        "topic": "sequences", "difficulty": "medium"
    },
    {
        "text": "Matrix multiplication: (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ. Number of columns in A must equal rows in B. Matrix A (m×n) times B (n×p) gives C (m×p). Matrix multiplication is not commutative: AB ≠ BA generally.",
        "topic": "linear_algebra", "difficulty": "hard"
    },
    {
        "text": "Determinant of 2×2 matrix: det([[a,b],[c,d]]) = ad-bc. Inverse of 2×2: A⁻¹ = (1/det(A))·[[d,-b],[-c,a]]. Matrix is invertible only if det(A) ≠ 0.",
        "topic": "linear_algebra", "difficulty": "hard"
    },
    {
        "text": "Limits: lim(x→a) f(x) = L means f(x) approaches L as x approaches a. L'Hôpital's rule: if limit gives 0/0 or ∞/∞, take derivative of numerator and denominator separately.",
        "topic": "calculus", "difficulty": "hard"
    },
    {
        "text": "Logarithm rules: log(ab) = log(a)+log(b). log(a/b) = log(a)-log(b). log(aⁿ) = n·log(a). log_b(b) = 1. log_b(1) = 0. Change of base: log_b(x) = ln(x)/ln(b).",
        "topic": "algebra", "difficulty": "medium"
    },
    {
        "text": "Area formulas: Circle = πr². Triangle = ½bh. Rectangle = lw. Trapezoid = ½(a+b)h. Volume formulas: Sphere = (4/3)πr³. Cylinder = πr²h. Cone = (1/3)πr²h. Cube = s³.",
        "topic": "geometry", "difficulty": "easy"
    },
    {
        "text": "Binomial theorem: (a+b)ⁿ = Σₖ₌₀ⁿ C(n,k)·aⁿ⁻ᵏ·bᵏ. First few expansions: (a+b)² = a²+2ab+b². (a+b)³ = a³+3a²b+3ab²+b³.",
        "topic": "algebra", "difficulty": "hard"
    },
    {
        "text": "Simple and compound interest: Simple interest I = PRT where P=principal, R=rate, T=time. Compound interest A = P(1+r/n)^(nt) where n=compounds per year. Continuous: A = Pe^(rt).",
        "topic": "applied_math", "difficulty": "easy"
    },
]


def main():
    print("Connecting to Qdrant...")
    rag = QdrantManager()

    print(f"Seeding {len(KNOWLEDGE_BASE)} documents...\n")
    for i, item in enumerate(KNOWLEDGE_BASE, 1):
        doc_id = rag.add_qa_pair(
            question = item["text"][:120],   # use start of text as the "question" for matching
            answer   = item["text"],
            source   = "seeded",
        )
        print(f"[{i:02d}/{len(KNOWLEDGE_BASE)}] ✓ {item['topic']:20s} ({item['difficulty']})  id={doc_id[:8]}...")

    total = rag.count()
    print(f"\nDone. Qdrant now has {total} documents.")


if __name__ == "__main__":
    main()