"""
AI-Powered Hiring Pipeline — Bias Detection Module
====================================================
Analyses candidate scoring for potential biases related to
gender, age, ethnicity, education prestige, and name bias.
Produces a structured BiasReport with flags and recommendations.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Reference lists for pattern matching (anonymised / statistical)
# ---------------------------------------------------------------------------

# Prestigious universities often over-weighted in hiring decisions
PRESTIGIOUS_UNIVERSITIES = {
    "stanford", "harvard", "mit", "yale", "princeton",
    "cambridge", "oxford", "caltech", "carnegie mellon",
    "uc berkeley", "columbia", "cornell", "upenn",
    "university of pennsylvania",
}

# Age-correlated phrases that could introduce bias
AGE_INDICATORS_YOUNG = [
    "recent graduate", "junior", "entry-level", "bootcamp",
    "1 year", "fresh grad", "intern",
]

AGE_INDICATORS_SENIOR = [
    "15+ years", "20 years", "veteran", "extensive career",
    "decades of experience",
]

# Education prestige bias detection
ELITE_EDUCATION_TERMS = [
    "ivy league", "top-tier university", "prestigious institution",
]


def detect_bias(
    candidate: dict,
    score: dict,
    job: dict,
) -> dict:
    """Analyse a candidate's scoring for potential biases.

    Args:
        candidate: Candidate data dict
        score: CandidateScore data dict
        job: JobDescription data dict

    Returns:
        A dict matching the BiasReport schema with flags, risk, and notes.
    """
    flags = []
    notes_parts = []
    candidate_id = candidate.get("id", "")
    overall_score = score.get("overall_score", 0.0)

    # ------------------------------------------------------------------
    # 1. Education Prestige Bias
    # ------------------------------------------------------------------
    university = (candidate.get("university") or "").lower()
    education_score = score.get("education_match", 0.0)

    is_prestigious = any(p in university for p in PRESTIGIOUS_UNIVERSITIES)
    if is_prestigious and education_score > 85:
        flags.append({
            "category": "education_prestige",
            "severity": "medium",
            "description": (
                f"Education score ({education_score:.0f}) may be inflated by "
                f"university prestige ({candidate.get('university', '')}). "
                f"Consider whether the education genuinely aligns with "
                f"job requirements."
            ),
            "recommendation": (
                "Evaluate education based on degree relevance, not "
                "institution ranking. A relevant degree from any "
                "accredited institution should score equally."
            ),
        })
        notes_parts.append("Prestigious university detected — check for prestige bias")

    # Non-traditional education should not be penalised
    education = (candidate.get("education") or "").lower()
    if ("bootcamp" in education or "certificate" in education) and education_score < 30:
        flags.append({
            "category": "education_prestige",
            "severity": "medium",
            "description": (
                f"Non-traditional education ({candidate.get('education', '')}) "
                f"scored very low ({education_score:.0f}). Bootcamp and "
                f"certificate graduates can be equally qualified."
            ),
            "recommendation": (
                "Focus on demonstrated skills and portfolio over "
                "formal education credentials. Consider practical "
                "experience as equivalent."
            ),
        })
        notes_parts.append("Non-traditional education potentially penalised")

    # ------------------------------------------------------------------
    # 2. Age Bias Indicators
    # ------------------------------------------------------------------
    summary = (candidate.get("summary") or "").lower()
    years_exp = candidate.get("years_experience", 0)

    # Young candidate bias
    young_indicators = [ind for ind in AGE_INDICATORS_YOUNG if ind in summary]
    if young_indicators and overall_score < 30:
        flags.append({
            "category": "age",
            "severity": "low",
            "description": (
                f"Young/junior indicators detected ({', '.join(young_indicators)}) "
                f"combined with low overall score ({overall_score:.0f}). "
                f"Verify score reflects skill assessment, not age assumptions."
            ),
            "recommendation": (
                "Evaluate based on demonstrated ability and growth "
                "potential rather than years of experience alone."
            ),
        })
        notes_parts.append("Possible age bias against junior candidate")

    # Overqualified / senior bias
    if years_exp > 12 and overall_score < 50:
        flags.append({
            "category": "age",
            "severity": "low",
            "description": (
                f"Highly experienced candidate ({years_exp} years) scored "
                f"below 50 ({overall_score:.0f}). This could indicate "
                f"'overqualified' bias or age-related assumptions."
            ),
            "recommendation": (
                "Assess whether lower score is due to genuine skill "
                "mismatch or unconscious bias about overqualification."
            ),
        })
        notes_parts.append("Senior candidate possibly penalised for experience level")

    # ------------------------------------------------------------------
    # 3. Name Bias Awareness
    # ------------------------------------------------------------------
    name = candidate.get("name", "")
    # We flag ANY name-based considerations — the system should be name-blind
    if name:
        flags.append({
            "category": "name_bias",
            "severity": "low",
            "description": (
                f"Candidate name '{name}' is visible during scoring. "
                f"Research shows name-based bias can affect hiring decisions. "
                f"In production, consider anonymising names during scoring."
            ),
            "recommendation": (
                "Implement name-blind scoring in production. Remove "
                "candidate names from the scoring prompt and reveal "
                "only during the final human review stage."
            ),
        })

    # ------------------------------------------------------------------
    # 4. Skills Gap vs Score Consistency
    # ------------------------------------------------------------------
    skills_match = score.get("skills_match", 0.0)
    experience_match = score.get("experience_match", 0.0)

    # Large discrepancy between skills and overall score
    if abs(skills_match - overall_score) > 30:
        flags.append({
            "category": "none",
            "severity": "low",
            "description": (
                f"Significant gap between skills match ({skills_match:.0f}) "
                f"and overall score ({overall_score:.0f}). This may indicate "
                f"inconsistent weighting or subjective bias in scoring."
            ),
            "recommendation": (
                "Ensure scoring weights are consistent and transparent. "
                "Document the rationale for any score that diverges "
                "significantly from individual component scores."
            ),
        })

    # ------------------------------------------------------------------
    # Determine overall risk level
    # ------------------------------------------------------------------
    high_flags = sum(1 for f in flags if f.get("severity") == "high")
    medium_flags = sum(1 for f in flags if f.get("severity") == "medium")

    if high_flags > 0:
        overall_risk = "high"
    elif medium_flags >= 2:
        overall_risk = "high"
    elif medium_flags == 1:
        overall_risk = "medium"
    elif len(flags) > 2:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    # ------------------------------------------------------------------
    # Adjusted score (remove prestige bias if detected)
    # ------------------------------------------------------------------
    adjusted_score = overall_score
    prestige_flags = [f for f in flags if f.get("category") == "education_prestige"]
    if prestige_flags and is_prestigious:
        # Reduce education weight by capping it to the average of other scores
        other_avg = (skills_match + experience_match) / 2
        if education_score > other_avg:
            adjustment = (education_score - other_avg) * 0.25 * 0.10
            adjusted_score = max(0, overall_score - adjustment)

    return {
        "candidate_id": candidate_id,
        "flags": flags,
        "overall_risk": overall_risk,
        "adjusted_score": round(adjusted_score, 1),
        "notes": "; ".join(notes_parts) if notes_parts else "No significant bias indicators",
    }
