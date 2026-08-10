"""
Procedural Memory & Self-Improving Agent — Evaluator & Learning Curve
=====================================================================
Computes learning metrics, generates learning curve charts,
and runs the 20-interaction evaluation demonstration.
"""

import os
import time
import json
from pathlib import Path

from models import LearningCurvePoint, PerformanceRecord
from memory import ProceduralMemoryStore
from agent import SelfImprovingAgent


# ═══════════════════════════════════════════════════════════════════════════
#  LEARNING CURVE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_learning_curve(memory: ProceduralMemoryStore) -> list[LearningCurvePoint]:
    """Compute the learning curve from performance records."""
    records = memory.get_performance_records()
    if not records:
        return []

    curve = []
    correct_count = 0

    for i, record in enumerate(records):
        if record.was_correct:
            correct_count += 1

        cumulative_accuracy = correct_count / (i + 1)

        # Rolling accuracy (window of 5)
        window_start = max(0, i - 4)
        window_records = records[window_start:i + 1]
        rolling_correct = sum(1 for r in window_records if r.was_correct)
        rolling_accuracy = rolling_correct / len(window_records)

        # Error rate
        error_rate = 1.0 - rolling_accuracy

        curve.append(LearningCurvePoint(
            interaction_number=record.interaction_number,
            cumulative_accuracy=round(cumulative_accuracy, 3),
            rolling_accuracy=round(rolling_accuracy, 3),
            total_rules=record.total_rules_available,
            error_rate=round(error_rate, 3),
        ))

    return curve


def generate_learning_curve_chart(memory: ProceduralMemoryStore, output_path: str = "learning_curve.png"):
    """Generate a matplotlib learning curve chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed. Skipping chart generation.")
        return

    curve = compute_learning_curve(memory)
    if not curve:
        print("No performance data to plot.")
        return

    interactions = [p.interaction_number for p in curve]
    cumulative_acc = [p.cumulative_accuracy for p in curve]
    rolling_acc = [p.rolling_accuracy for p in curve]
    error_rates = [p.error_rate for p in curve]
    total_rules = [p.total_rules for p in curve]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
    fig.suptitle("Self-Improving Agent — Learning Curve", fontsize=16, fontweight="bold", color="#2c3e50")
    fig.patch.set_facecolor("#f8f9fa")

    # ── Top chart: Accuracy ──
    ax1.set_facecolor("#ffffff")
    ax1.plot(interactions, cumulative_acc, "o-", color="#2ecc71", linewidth=2, markersize=5,
             label="Cumulative Accuracy", alpha=0.9)
    ax1.plot(interactions, rolling_acc, "s-", color="#3498db", linewidth=2, markersize=5,
             label="Rolling Accuracy (window=5)", alpha=0.9)
    ax1.fill_between(interactions, rolling_acc, alpha=0.1, color="#3498db")
    ax1.set_ylabel("Accuracy", fontsize=12, color="#2c3e50")
    ax1.set_ylim(-0.05, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy Over Time", fontsize=13, color="#2c3e50")

    # ── Bottom chart: Error Rate + Rules ──
    ax2.set_facecolor("#ffffff")
    ax2_twin = ax2.twinx()

    ax2.bar(interactions, error_rates, color="#e74c3c", alpha=0.6, label="Error Rate", width=0.8)
    ax2_twin.plot(interactions, total_rules, "D-", color="#f39c12", linewidth=2, markersize=5,
                  label="Total Rules", alpha=0.9)

    ax2.set_xlabel("Interaction Number", fontsize=12, color="#2c3e50")
    ax2.set_ylabel("Error Rate", fontsize=12, color="#e74c3c")
    ax2_twin.set_ylabel("Total Rules", fontsize=12, color="#f39c12")
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Error Rate & Rule Growth", fontsize=13, color="#2c3e50")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Learning curve chart saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  20-INTERACTION EVALUATION DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

# Predefined evaluation scenarios with deliberate mistakes and corrections
EVAL_SCENARIOS = [
    # ── Interactions 1-5: Agent makes mistakes, user corrects ──
    {
        "question": "What is the capital of Australia?",
        "correction": "The capital of Australia is Canberra, not Sydney. Many people confuse this, but Canberra has been the capital since 1913.",
        "expected_rule_domain": "factual",
    },
    {
        "question": "Explain quantum computing in one paragraph.",
        "correction": "Your explanation was too technical and used jargon without defining terms. When explaining complex topics, start with an analogy and define technical terms before using them.",
        "expected_rule_domain": "formatting",
    },
    {
        "question": "What is the time complexity of binary search?",
        "correction": "The time complexity is O(log n), not O(n log n). O(n log n) is merge sort. Be precise about algorithmic complexities.",
        "expected_rule_domain": "accuracy",
    },
    {
        "question": "How does photosynthesis work?",
        "correction": "You forgot to mention the light-independent reactions (Calvin cycle). Always cover both the light-dependent reactions AND the Calvin cycle when explaining photosynthesis.",
        "expected_rule_domain": "completeness",
    },
    {
        "question": "Write a Python function to reverse a string.",
        "correction": "Your solution used slicing [::-1] which is fine, but you should also mention that this creates a new string and doesn't modify in-place, since Python strings are immutable. Always clarify mutability when discussing string operations.",
        "expected_rule_domain": "completeness",
    },

    # ── Interactions 6-10: Mix of new topics and topics where rules should apply ──
    {
        "question": "What is the capital of Brazil?",
        "correction": None,  # Agent should get this right (learned from Australia correction)
    },
    {
        "question": "Explain machine learning in one paragraph.",
        "correction": None,  # Should use simple language (learned from quantum computing correction)
    },
    {
        "question": "What is the time complexity of merge sort?",
        "correction": None,  # Should be precise (learned from binary search correction)
    },
    {
        "question": "How does cellular respiration work?",
        "correction": "You covered glycolysis and the Krebs cycle but didn't mention the electron transport chain. The same completeness issue as before — cover ALL major stages.",
        "expected_rule_domain": "completeness",
    },
    {
        "question": "Write a Python function to remove duplicates from a list.",
        "correction": None,  # Should mention mutability considerations (learned from string reversal)
    },

    # ── Interactions 11-15: Agent should be improving ──
    {
        "question": "What is the capital of Turkey?",
        "correction": None,  # Should know it's Ankara, not Istanbul
    },
    {
        "question": "Explain blockchain in one paragraph.",
        "correction": None,  # Should be accessible
    },
    {
        "question": "What is the time complexity of quicksort?",
        "correction": "You said O(n log n) which is the average case, but you didn't mention that the worst case is O(n²). Always specify both average and worst case for sorting algorithms.",
        "expected_rule_domain": "accuracy",
    },
    {
        "question": "How does the water cycle work?",
        "correction": None,  # Should cover all phases
    },
    {
        "question": "Write a Python function to flatten a nested list.",
        "correction": None,  # Should discuss mutability/copying
    },

    # ── Interactions 16-20: Agent should be mostly correct ──
    {
        "question": "What is the capital of Myanmar?",
        "correction": None,  # Naypyidaw, not Yangon
    },
    {
        "question": "Explain neural networks in one paragraph.",
        "correction": None,  # Accessible language
    },
    {
        "question": "What is the time complexity of heap sort?",
        "correction": None,  # Should give both average and worst case
    },
    {
        "question": "Describe the nitrogen cycle.",
        "correction": None,  # Should cover all stages
    },
    {
        "question": "Write a Python function to merge two sorted lists.",
        "correction": None,  # Should discuss in-place vs new list
    },
]


def run_evaluation(db_path: str = "self_improving_agent_eval.db") -> dict:
    """Run the full 20-interaction evaluation demonstration.

    Returns a dict with evaluation results and metrics.
    """
    # Start fresh for evaluation
    if os.path.exists(db_path):
        os.remove(db_path)

    agent = SelfImprovingAgent(db_path=db_path)

    results = {
        "interactions": [],
        "corrections_applied": 0,
        "rules_extracted": 0,
        "early_error_rate": 0.0,
        "late_error_rate": 0.0,
    }

    print("=" * 70)
    print("  Self-Improving Agent — 20-Interaction Evaluation")
    print("=" * 70)

    for i, scenario in enumerate(EVAL_SCENARIOS, 1):
        print(f"\n--- Interaction {i}/20 ---")
        print(f"Q: {scenario['question']}")

        # Get response
        response, applied_rules = agent.respond(scenario["question"])
        print(f"A: {response[:150]}...")

        if applied_rules:
            print(f"   [Applied {len(applied_rules)} rules]")

        interaction_result = {
            "number": i,
            "question": scenario["question"],
            "response": response[:200],
            "rules_applied": len(applied_rules),
            "was_corrected": False,
        }

        # Handle correction if applicable
        if scenario.get("correction"):
            print(f"   CORRECTION: {scenario['correction'][:100]}...")
            rule = agent.handle_correction(
                scenario["question"], response, scenario["correction"]
            )
            results["corrections_applied"] += 1
            results["rules_extracted"] += 1
            interaction_result["was_corrected"] = True
            interaction_result["rule_extracted"] = rule.rule_text
            print(f"   RULE EXTRACTED: {rule.rule_text[:100]}")

        results["interactions"].append(interaction_result)
        time.sleep(2)  # Rate limit

    # Compute metrics
    memory = ProceduralMemoryStore(db_path=db_path)

    # Early error rate (interactions 1-5)
    early_errors = sum(1 for r in results["interactions"][:5] if r["was_corrected"])
    results["early_error_rate"] = early_errors / 5

    # Late error rate (interactions 16-20)
    late_errors = sum(1 for r in results["interactions"][15:] if r["was_corrected"])
    results["late_error_rate"] = late_errors / 5

    # Overall stats
    results["total_rules"] = memory.count_rules()
    results["improvement"] = results["early_error_rate"] - results["late_error_rate"]
    results["learning_curve"] = [p.model_dump() for p in compute_learning_curve(memory)]

    # Generate chart
    chart_path = str(Path(db_path).parent / "learning_curve.png")
    generate_learning_curve_chart(memory, output_path=chart_path)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total interactions: 20")
    print(f"  Total corrections:  {results['corrections_applied']}")
    print(f"  Total rules:        {results['total_rules']}")
    print(f"  Early error rate (1-5):   {results['early_error_rate']:.0%}")
    print(f"  Late error rate (16-20):  {results['late_error_rate']:.0%}")
    print(f"  Improvement:              {results['improvement']:.0%}")
    print(f"  Learning curve chart:     {chart_path}")

    return results


if __name__ == "__main__":
    eval_results = run_evaluation()

    # Save evaluation report
    report_path = Path(__file__).parent / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"\nEvaluation report saved to {report_path}")
