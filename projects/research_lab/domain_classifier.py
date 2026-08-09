"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — Domain Classifier

Classifies research topics into domains and dynamically assembles
the optimal specialist agent team (3–5 agents) for each domain.
This is the "brain" that makes the system adaptive rather than fixed.
"""

import os
import json
import re
import time

from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

from projects.research_lab.models import ResearchDomain, AgentRole

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


def _llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Make a single LLM call via Groq with retry."""
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 12
                time.sleep(wait)
            else:
                raise
    return "general"


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling markdown code blocks."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  DOMAIN → AGENT TEAM MAPPING
# ═══════════════════════════════════════════════════════════════════════════

# Each domain gets a curated team of specialist agents. The key insight is
# that different research domains need different expertise combinations.
# Medical research needs methodology review + ethics considerations,
# while tech research needs data analysis + implementation feasibility.

DOMAIN_AGENT_TEAMS: dict[str, list[dict]] = {
    ResearchDomain.TECHNOLOGY: [
        {
            "role": AgentRole.HYPOTHESIS_GENERATOR,
            "name": "tech-hypothesis-agent",
            "description": "Generates testable hypotheses about technology trends, adoption, and impact",
            "system_prompt": (
                "You are a technology research hypothesis generator. Given a research topic, "
                "generate 3 novel, testable hypotheses. Each hypothesis must be specific, "
                "falsifiable, and grounded in current technology trends. Consider: technical "
                "feasibility, market adoption patterns, competitive dynamics, and societal impact."
            ),
        },
        {
            "role": AgentRole.LITERATURE_REVIEWER,
            "name": "tech-literature-agent",
            "description": "Reviews existing research, papers, and industry reports",
            "system_prompt": (
                "You are a technology literature review specialist. Summarise the existing "
                "body of knowledge on the given topic. Cover: seminal papers, recent "
                "developments, industry reports, open-source ecosystem trends, and "
                "identified research gaps. Be specific about evidence quality."
            ),
        },
        {
            "role": AgentRole.DATA_ANALYST,
            "name": "tech-data-agent",
            "description": "Analyses quantitative data, benchmarks, and metrics",
            "system_prompt": (
                "You are a technology data analyst. Provide quantitative evidence: "
                "market sizing data, benchmark comparisons, adoption rate statistics, "
                "performance metrics, and cost-benefit analyses. Cite specific numbers."
            ),
        },
        {
            "role": AgentRole.DOMAIN_SPECIALIST,
            "name": "tech-specialist-agent",
            "description": "Deep domain expertise in the specific technology area",
            "system_prompt": (
                "You are a senior technology specialist. Provide deep domain expertise: "
                "architectural trade-offs, implementation challenges, security implications, "
                "scalability concerns, and comparison with alternative approaches. "
                "Draw on practical engineering experience."
            ),
        },
    ],
    ResearchDomain.MEDICINE: [
        {
            "role": AgentRole.HYPOTHESIS_GENERATOR,
            "name": "med-hypothesis-agent",
            "description": "Generates hypotheses grounded in medical science",
            "system_prompt": (
                "You are a medical research hypothesis generator. Generate 3 testable "
                "hypotheses about the given medical/health topic. Each hypothesis must "
                "be clinically relevant, ethically sound, and grounded in existing "
                "pathophysiology or epidemiological data."
            ),
        },
        {
            "role": AgentRole.LITERATURE_REVIEWER,
            "name": "med-literature-agent",
            "description": "Reviews clinical trials, meta-analyses, and medical literature",
            "system_prompt": (
                "You are a medical literature review specialist. Summarise relevant "
                "clinical trials, systematic reviews, meta-analyses, and treatment "
                "guidelines. Assess evidence levels (RCT, cohort, case study). "
                "Note conflicting findings and methodological concerns."
            ),
        },
        {
            "role": AgentRole.METHODOLOGY_EXPERT,
            "name": "med-methodology-agent",
            "description": "Evaluates study design, statistical methods, and ethical standards",
            "system_prompt": (
                "You are a medical research methodology expert. Evaluate: appropriate "
                "study design (RCT vs observational), sample size adequacy, statistical "
                "methods, control for confounders, and ethical considerations (IRB, "
                "informed consent, vulnerable populations)."
            ),
        },
        {
            "role": AgentRole.DOMAIN_SPECIALIST,
            "name": "med-specialist-agent",
            "description": "Clinical domain expertise",
            "system_prompt": (
                "You are a clinical specialist. Provide domain expertise: "
                "pathophysiology mechanisms, differential diagnoses, treatment "
                "options, drug interactions, contraindications, and patient outcomes "
                "data. Focus on clinical applicability."
            ),
        },
        {
            "role": AgentRole.DATA_ANALYST,
            "name": "med-data-agent",
            "description": "Analyses epidemiological data and clinical statistics",
            "system_prompt": (
                "You are a medical data analyst. Provide quantitative evidence: "
                "prevalence/incidence rates, odds ratios, number needed to treat (NNT), "
                "survival curves, dose-response data, and demographic breakdowns."
            ),
        },
    ],
    ResearchDomain.ECONOMICS: [
        {
            "role": AgentRole.HYPOTHESIS_GENERATOR,
            "name": "econ-hypothesis-agent",
            "description": "Generates economic hypotheses with theoretical grounding",
            "system_prompt": (
                "You are an economics research hypothesis generator. Generate 3 testable "
                "hypotheses grounded in economic theory. Consider: market structures, "
                "behavioral economics, macroeconomic indicators, trade dynamics, and "
                "policy implications. Each must be empirically testable."
            ),
        },
        {
            "role": AgentRole.LITERATURE_REVIEWER,
            "name": "econ-literature-agent",
            "description": "Reviews economic papers, policy reports, and empirical studies",
            "system_prompt": (
                "You are an economics literature reviewer. Summarise: seminal papers, "
                "recent empirical studies, central bank reports, IMF/World Bank data, "
                "and policy analysis. Note methodological approaches (econometric models, "
                "natural experiments, difference-in-differences)."
            ),
        },
        {
            "role": AgentRole.DATA_ANALYST,
            "name": "econ-data-agent",
            "description": "Analyses economic data, indicators, and statistical models",
            "system_prompt": (
                "You are an economic data analyst. Provide: GDP/CPI/unemployment data, "
                "regression results, Gini coefficients, trade balances, elasticities, "
                "and forecasting models. Be specific about statistical significance."
            ),
        },
        {
            "role": AgentRole.DOMAIN_SPECIALIST,
            "name": "econ-specialist-agent",
            "description": "Deep expertise in the specific economic sub-field",
            "system_prompt": (
                "You are an economics specialist. Provide expert analysis: market "
                "equilibrium dynamics, regulatory impact assessment, behavioral biases, "
                "game-theoretic implications, and policy recommendation frameworks."
            ),
        },
    ],
    ResearchDomain.ENVIRONMENT: [
        {
            "role": AgentRole.HYPOTHESIS_GENERATOR,
            "name": "env-hypothesis-agent",
            "description": "Generates environmental science hypotheses",
            "system_prompt": (
                "You are an environmental research hypothesis generator. Generate 3 "
                "hypotheses about the environmental topic. Consider: climate models, "
                "ecosystem dynamics, biodiversity metrics, pollution pathways, and "
                "sustainability transitions. Each must be measurable."
            ),
        },
        {
            "role": AgentRole.LITERATURE_REVIEWER,
            "name": "env-literature-agent",
            "description": "Reviews environmental studies, IPCC reports, and field data",
            "system_prompt": (
                "You are an environmental literature reviewer. Summarise: IPCC "
                "reports, nature/science publications, EPA data, satellite imagery "
                "studies, and long-term monitoring datasets. Note confidence levels "
                "and model uncertainties."
            ),
        },
        {
            "role": AgentRole.DATA_ANALYST,
            "name": "env-data-agent",
            "description": "Analyses environmental metrics and climate data",
            "system_prompt": (
                "You are an environmental data analyst. Provide: temperature trends, "
                "CO2 concentrations, deforestation rates, species population data, "
                "water quality indices, and emissions projections with uncertainty bounds."
            ),
        },
        {
            "role": AgentRole.METHODOLOGY_EXPERT,
            "name": "env-methodology-agent",
            "description": "Evaluates environmental study design and modeling approaches",
            "system_prompt": (
                "You are an environmental research methodology expert. Evaluate: "
                "sampling strategies, remote sensing accuracy, GCM model assumptions, "
                "temporal coverage, spatial resolution, and reproducibility of findings."
            ),
        },
    ],
    ResearchDomain.SOCIAL_SCIENCE: [
        {
            "role": AgentRole.HYPOTHESIS_GENERATOR,
            "name": "soc-hypothesis-agent",
            "description": "Generates social science hypotheses",
            "system_prompt": (
                "You are a social science hypothesis generator. Generate 3 testable "
                "hypotheses. Consider: sociological theories, psychological frameworks, "
                "cultural dynamics, institutional effects, and demographic trends. "
                "Each must be operationalisable and measurable."
            ),
        },
        {
            "role": AgentRole.LITERATURE_REVIEWER,
            "name": "soc-literature-agent",
            "description": "Reviews social science literature and surveys",
            "system_prompt": (
                "You are a social science literature reviewer. Summarise: foundational "
                "theories, recent empirical studies, survey data, longitudinal studies, "
                "and cross-cultural comparisons. Note sample sizes and generalisability."
            ),
        },
        {
            "role": AgentRole.METHODOLOGY_EXPERT,
            "name": "soc-methodology-agent",
            "description": "Evaluates survey design, sampling, and qualitative methods",
            "system_prompt": (
                "You are a social science methodology expert. Evaluate: survey design, "
                "sampling bias, response rates, qualitative coding reliability, "
                "mixed methods integration, and ethical considerations (IRB, "
                "informed consent, deception)."
            ),
        },
        {
            "role": AgentRole.DATA_ANALYST,
            "name": "soc-data-agent",
            "description": "Analyses survey data, demographics, and social indicators",
            "system_prompt": (
                "You are a social science data analyst. Provide: survey results, "
                "regression analyses, effect sizes (Cohen's d), demographic breakdowns, "
                "longitudinal trends, and cross-cultural comparison data."
            ),
        },
    ],
}

# General domain is a fallback using a balanced team
DOMAIN_AGENT_TEAMS[ResearchDomain.GENERAL] = [
    {
        "role": AgentRole.HYPOTHESIS_GENERATOR,
        "name": "gen-hypothesis-agent",
        "description": "Generates hypotheses for any domain",
        "system_prompt": (
            "You are a general research hypothesis generator. Generate 3 testable "
            "hypotheses about the given topic. Be specific, falsifiable, and grounded "
            "in existing knowledge. Consider multiple disciplinary perspectives."
        ),
    },
    {
        "role": AgentRole.LITERATURE_REVIEWER,
        "name": "gen-literature-agent",
        "description": "General literature review across disciplines",
        "system_prompt": (
            "You are a general literature reviewer. Provide a comprehensive review "
            "of existing research, spanning multiple disciplines as appropriate. "
            "Note key findings, consensus views, and active debates."
        ),
    },
    {
        "role": AgentRole.DATA_ANALYST,
        "name": "gen-data-agent",
        "description": "General quantitative analysis",
        "system_prompt": (
            "You are a general data analyst. Provide relevant quantitative evidence: "
            "statistics, trends, comparisons, and benchmarks from reliable sources."
        ),
    },
]

# Agents that are ALWAYS included regardless of domain
UNIVERSAL_AGENTS = [
    {
        "role": AgentRole.CRITIC,
        "name": "critic-agent",
        "description": "Challenges hypotheses, evidence quality, and methodology — the adversarial reviewer",
        "system_prompt": (
            "You are a rigorous research critic. Your job is to find weaknesses, "
            "flaws, and gaps in the research. Challenge: hypothesis novelty, "
            "evidence quality, methodology soundness, logical consistency, and "
            "potential biases. Be constructive but thorough. Score issues by severity."
        ),
    },
    {
        "role": AgentRole.SYNTHESISER,
        "name": "synthesis-agent",
        "description": "Merges all agent findings into a coherent research paper",
        "system_prompt": (
            "You are a research synthesis specialist. Take findings from multiple "
            "agents and produce a coherent, well-structured research paper with: "
            "abstract, introduction, methodology, findings, discussion, limitations, "
            "and conclusion. Resolve contradictions, attribute contributions."
        ),
    },
    {
        "role": AgentRole.PEER_REVIEWER,
        "name": "peer-reviewer-agent",
        "description": "Final quality assessment — simulates academic peer review",
        "system_prompt": (
            "You are an academic peer reviewer. Evaluate the research paper for: "
            "originality, methodological rigor, evidence quality, logical consistency, "
            "clarity of writing, and contribution to the field. Provide a verdict "
            "(accept / minor revisions / major revisions / reject) with specific, "
            "actionable feedback."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  DOMAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class DomainClassifier:
    """
    Classifies a research topic into a domain and assembles the optimal
    agent team. Uses LLM for classification with keyword fallback.
    """

    # Keyword-based fallback when LLM is unavailable
    KEYWORD_MAP: dict[str, list[str]] = {
        ResearchDomain.TECHNOLOGY: [
            "ai", "machine learning", "software", "algorithm", "computing",
            "blockchain", "cybersecurity", "cloud", "data science", "robotics",
            "neural network", "llm", "api", "programming", "database",
            "quantum computing", "iot", "devops", "microservice",
        ],
        ResearchDomain.MEDICINE: [
            "health", "medical", "clinical", "disease", "drug", "therapy",
            "patient", "cancer", "vaccine", "diagnosis", "treatment",
            "pharmaceutical", "surgery", "epidemiol", "pathology", "genomic",
        ],
        ResearchDomain.ECONOMICS: [
            "economy", "economic", "market", "gdp", "inflation", "trade",
            "fiscal", "monetary", "banking", "finance", "investment",
            "cryptocurrency", "stock", "supply chain", "pricing",
        ],
        ResearchDomain.ENVIRONMENT: [
            "climate", "environment", "sustainability", "carbon", "emission",
            "pollution", "biodiversity", "ecosystem", "renewable", "energy",
            "deforestation", "ocean", "glacier", "warming", "conservation",
        ],
        ResearchDomain.SOCIAL_SCIENCE: [
            "social", "society", "culture", "education", "psychology",
            "behavior", "inequality", "gender", "race", "political",
            "demographic", "migration", "poverty", "urbanization",
        ],
    }

    def classify(self, topic: str) -> ResearchDomain:
        """Classify a research topic into a domain. LLM-first, keyword fallback."""
        # Try LLM classification first
        try:
            return self._classify_with_llm(topic)
        except Exception:
            return self._classify_with_keywords(topic)

    def _classify_with_llm(self, topic: str) -> ResearchDomain:
        """Use LLM to classify the domain."""
        prompt = (
            "Classify this research topic into EXACTLY ONE domain.\n\n"
            "Valid domains: technology, medicine, economics, environment, social_science, general\n\n"
            'Respond in JSON (no markdown): {"domain": "<domain_name>"}\n\n'
            "Rules:\n"
            "- Choose the MOST specific domain that fits\n"
            '- Use "general" only if no specific domain applies\n'
            "- Be case-sensitive with the domain names"
        )
        raw = _llm_call(prompt, f"Topic: {topic}", temperature=0.1)
        data = _parse_json(raw)
        domain_str = str(data.get("domain", "general")).lower().strip()

        # Validate
        valid_domains = [d.value for d in ResearchDomain]
        if domain_str in valid_domains:
            return ResearchDomain(domain_str)
        return ResearchDomain.GENERAL

    def _classify_with_keywords(self, topic: str) -> ResearchDomain:
        """Fallback: keyword-based classification."""
        topic_lower = topic.lower()
        scores: dict[str, int] = {}

        for domain, keywords in self.KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in topic_lower)
            scores[domain] = score

        best_domain = max(scores, key=scores.get)  # type: ignore
        if scores[best_domain] > 0:
            return ResearchDomain(best_domain)
        return ResearchDomain.GENERAL

    def assemble_team(self, domain: ResearchDomain) -> list[dict]:
        """
        Assemble the optimal agent team for a given domain.
        Returns 3–5 domain specialists + 3 universal agents = 6–8 total.
        """
        domain_agents = DOMAIN_AGENT_TEAMS.get(domain, DOMAIN_AGENT_TEAMS[ResearchDomain.GENERAL])
        full_team = domain_agents + UNIVERSAL_AGENTS
        return full_team

    def classify_and_assemble(self, topic: str) -> tuple[ResearchDomain, list[dict]]:
        """Convenience: classify the topic and assemble the team in one call."""
        domain = self.classify(topic)
        team = self.assemble_team(domain)
        return domain, team
