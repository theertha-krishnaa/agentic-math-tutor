"""
eval/run_eval.py
────────────────
RAGAS-style evaluation of the math tutor pipeline.
Runs every question in the golden dataset through the full pipeline
and measures: faithfulness, answer correctness, retrieval hit rate,
refinement rate, guardrail rejection rate, and latency.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --limit 20        # quick smoke test
    python eval/run_eval.py --output results.json
"""

import sys
import os
import json
import time
import argparse
import re
from pathlib import Path
from tqdm import tqdm

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from guardrails import validate_input
from agent import MCPRouter


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run RAGAS evaluation")
    p.add_argument("--limit",  type=int, default=None, help="Limit number of questions (for quick tests)")
    p.add_argument("--output", type=str, default="eval/latest_results.json", help="Output file for results")
    p.add_argument("--dataset", type=str, default="eval/golden_dataset.json", help="Path to golden dataset")
    return p.parse_args()


# ── Faithfulness scoring ──────────────────────────────────────────────────────
def score_faithfulness(generated: str, expected: str) -> float:
    """
    Measures whether the generated answer contains the key information
    from the expected answer.

    Uses a combination of:
    - Numeric value matching (most important for math)
    - Key term overlap
    Returns a score between 0.0 and 1.0
    """
    gen   = generated.lower()
    exp   = expected.lower()

    # Extract all numbers from both answers
    gen_nums = set(re.findall(r"-?\d+\.?\d*", gen))
    exp_nums = set(re.findall(r"-?\d+\.?\d*", exp))

    # Numeric match — most critical for math
    if exp_nums:
        matched_nums = exp_nums.intersection(gen_nums)
        num_score    = len(matched_nums) / len(exp_nums)
    else:
        num_score = 1.0   # no numbers to check

    # Key term overlap — ignore stopwords
    stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "for",
                 "and", "or", "with", "at", "by", "from", "that", "this",
                 "it", "be", "as", "on", "was", "not", "but", "have"}

    gen_terms  = set(gen.split()) - stopwords
    exp_terms  = set(exp.split()) - stopwords
    if exp_terms:
        term_score = len(gen_terms & exp_terms) / len(exp_terms)
    else:
        term_score = 1.0

    # Weighted: numeric accuracy matters more for math
    return round(0.7 * num_score + 0.3 * term_score, 3)


def score_correctness(generated: str, expected: str) -> float:
    """
    Stricter check — does the final numerical answer match?
    Returns 1.0, 0.5, or 0.0.
    """
    gen_nums = re.findall(r"-?\d+\.?\d*", generated.lower())
    exp_nums = re.findall(r"-?\d+\.?\d*", expected.lower())

    if not exp_nums:
        # No numbers — check for key phrase match
        exp_words = set(expected.lower().split())
        gen_words = set(generated.lower().split())
        overlap   = len(exp_words & gen_words) / max(len(exp_words), 1)
        return round(min(overlap * 1.5, 1.0), 3)

    # Check if all expected numbers appear in the generated answer
    matched = sum(1 for n in exp_nums if n in gen_nums)
    return round(matched / len(exp_nums), 3)


# ── Main eval loop ────────────────────────────────────────────────────────────
def run_evaluation(dataset_path: str, limit: int = None, output_path: str = None):

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    print(f"\n{'='*60}")
    print(f"  MATH TUTOR — RAGAS EVALUATION")
    print(f"  Dataset: {dataset_path}")
    print(f"  Questions: {len(dataset)}")
    print(f"{'='*60}\n")

    # Initialise router
    print("Loading pipeline...")
    router = MCPRouter()
    print("Pipeline ready.\n")

    # Tracking metrics
    results        = []
    faithfulness   = []
    correctness    = []
    latencies      = []
    sources        = {"vector_db": 0, "llm_generated": 0, "llm_refined": 0, "llm_only": 0}
    refinements    = 0
    guardrail_hits = 0
    errors         = 0

    # Test guardrail false positive rate — these should all PASS (be allowed)
    guardrail_false_positives = 0
    for item in dataset:
        check = validate_input(item["question"])
        if not check["allowed"]:
            guardrail_false_positives += 1

    # Run eval
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        expected = item["answer"]

        t0 = time.perf_counter()
        try:
            result  = router.route(question)
            elapsed = time.perf_counter() - t0

            generated = result.get("answer", "")
            source    = result.get("source", "unknown")
            refined   = result.get("refinement_applied", False)

            faith   = score_faithfulness(generated, expected)
            correct = score_correctness(generated, expected)

            faithfulness.append(faith)
            correctness.append(correct)
            latencies.append(elapsed)
            sources[source] = sources.get(source, 0) + 1
            if refined:
                refinements += 1

            results.append({
                "question":    question,
                "expected":    expected,
                "generated":   generated[:300],
                "source":      source,
                "faithfulness": faith,
                "correctness":  correct,
                "confidence":  result.get("confidence", 0),
                "latency_s":   round(elapsed, 3),
                "refined":     refined,
                "topic":       item.get("topic", ""),
                "difficulty":  item.get("difficulty", ""),
            })

        except Exception as e:
            errors += 1
            latencies.append(time.perf_counter() - t0)
            results.append({
                "question":    question,
                "expected":    expected,
                "generated":   f"ERROR: {e}",
                "source":      "error",
                "faithfulness": 0.0,
                "correctness":  0.0,
                "confidence":  0.0,
                "latency_s":   0.0,
                "refined":     False,
                "topic":       item.get("topic", ""),
                "difficulty":  item.get("difficulty", ""),
            })

    # ── Compute summary metrics ───────────────────────────────────────────────
    n = len(dataset)
    avg_faith   = round(sum(faithfulness) / n, 4) if faithfulness else 0
    avg_correct = round(sum(correctness)  / n, 4) if correctness  else 0
    avg_latency = round(sum(latencies)    / n, 3) if latencies     else 0
    p95_latency = round(sorted(latencies)[int(0.95 * len(latencies))], 3) if latencies else 0

    db_hit_rate        = round(sources.get("vector_db", 0) / n, 3)
    llm_rate           = round((sources.get("llm_generated", 0) + sources.get("llm_only", 0)) / n, 3)
    refinement_rate    = round(refinements / n, 3)
    fp_rate            = round(guardrail_false_positives / n, 3)
    error_rate         = round(errors / n, 3)

    summary = {
        "total_questions":         n,
        "avg_faithfulness":        avg_faith,
        "avg_answer_correctness":  avg_correct,
        "retrieval_hit_rate":      db_hit_rate,
        "llm_call_rate":           llm_rate,
        "refinement_rate":         refinement_rate,
        "guardrail_false_positive_rate": fp_rate,
        "error_rate":              error_rate,
        "avg_latency_s":           avg_latency,
        "p95_latency_s":           p95_latency,
        "source_breakdown":        sources,
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Questions evaluated:     {n}")
    print(f"  Avg faithfulness:        {avg_faith:.1%}")
    print(f"  Avg answer correctness:  {avg_correct:.1%}")
    print(f"  Retrieval hit rate:      {db_hit_rate:.1%}  (DB hits, no LLM needed)")
    print(f"  LLM call rate:           {llm_rate:.1%}")
    print(f"  Refinement rate:         {refinement_rate:.1%}")
    print(f"  Guardrail false pos:     {fp_rate:.1%}  (math Q incorrectly blocked)")
    print(f"  Error rate:              {error_rate:.1%}")
    print(f"  Avg latency:             {avg_latency:.2f}s")
    print(f"  P95 latency:             {p95_latency:.2f}s")
    print(f"\n  Source breakdown:")
    for src, count in sources.items():
        print(f"    {src:<20} {count:>4}  ({count/n:.1%})")

    # Per-topic breakdown
    topic_scores = {}
    for r in results:
        t = r.get("topic", "unknown")
        if t not in topic_scores:
            topic_scores[t] = []
        topic_scores[t].append(r["correctness"])

    print(f"\n  Correctness by topic:")
    for topic, scores in sorted(topic_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"    {topic:<20} {avg:.1%}  ({len(scores)} questions)")

    print(f"\n{'='*60}\n")

    # ── Save output ───────────────────────────────────────────────────────────
    if output_path:
        output = {"summary": summary, "results": results}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Full results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    args    = parse_args()
    summary = run_evaluation(
        dataset_path = args.dataset,
        limit        = args.limit,
        output_path  = args.output,
    )

    # Exit with error code if metrics are below thresholds
    # (used by GitHub Actions CI on Day 2)
    FAITHFULNESS_THRESHOLD = 0.60
    CORRECTNESS_THRESHOLD  = 0.55

    if summary["avg_faithfulness"] < FAITHFULNESS_THRESHOLD:
        print(f"FAIL: Faithfulness {summary['avg_faithfulness']:.1%} below threshold {FAITHFULNESS_THRESHOLD:.1%}")
        sys.exit(1)
    if summary["avg_answer_correctness"] < CORRECTNESS_THRESHOLD:
        print(f"FAIL: Correctness {summary['avg_answer_correctness']:.1%} below threshold {CORRECTNESS_THRESHOLD:.1%}")
        sys.exit(1)

    print("PASS: All metrics above thresholds.")
    sys.exit(0)