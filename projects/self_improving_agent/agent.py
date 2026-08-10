"""
Procedural Memory & Self-Improving Agent — Core Agent
======================================================
Groq-powered agent with procedural memory integration.
Extracts correction rules from user feedback and applies
them prospectively to prevent repeated mistakes.
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq

from models import CorrectionRule, InteractionLog, PerformanceRecord, RuleDomain
from memory import ProceduralMemoryStore

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Make a single LLM call via Groq with retry logic."""
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
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 12
                time.sleep(wait)
            else:
                raise
    return "Unable to generate response after retries."


class SelfImprovingAgent:
    """An agent that learns from corrections and improves over time."""

    def __init__(self, db_path: str = "self_improving_agent.db"):
        self.memory = ProceduralMemoryStore(db_path=db_path)
        self.interaction_count = self.memory.count_interactions()

    def respond(self, user_message: str) -> tuple[str, list[str]]:
        """Generate a response with rule-augmented prompting.

        Returns:
            Tuple of (response_text, list_of_applied_rule_ids)
        """
        # 1. Retrieve relevant rules
        relevant_rules = self._retrieve_relevant_rules(user_message)
        applied_rule_ids = [r.rule_id for r in relevant_rules]

        # 2. Build rule-augmented system prompt
        system_prompt = self._build_system_prompt(relevant_rules)

        # 3. Generate response
        response = llm_call(system_prompt, user_message, temperature=0.7)

        # 4. Increment application counts for used rules
        for rule in relevant_rules:
            self.memory.increment_rule_application(rule.rule_id)

        # 5. Log interaction (without correction initially)
        self.interaction_count += 1
        interaction = InteractionLog(
            user_message=user_message,
            agent_response=response,
            was_corrected=False,
            rules_applied=applied_rule_ids,
        )
        self.memory.log_interaction(interaction)

        # 6. Record performance (assume correct unless corrected)
        perf = PerformanceRecord(
            interaction_number=self.interaction_count,
            was_correct=True,
            rules_applied_count=len(applied_rule_ids),
            total_rules_available=self.memory.count_rules(),
            quality_score=0.7 + 0.05 * len(applied_rule_ids),  # Baseline + rule bonus
        )
        self.memory.record_performance(perf)

        return response, applied_rule_ids

    def handle_correction(self, user_message: str, original_response: str, correction: str) -> CorrectionRule:
        """Process a user correction: extract a rule and store it.

        Returns:
            The extracted CorrectionRule
        """
        # 1. Extract a generalised rule from the correction
        rule = self._extract_rule(user_message, original_response, correction)

        # 2. Store the rule
        self.memory.store_rule(rule)

        # 3. Update the last interaction as corrected
        recent = self.memory.get_recent_interactions(limit=1)
        if recent:
            corrected_interaction = recent[0]
            corrected_interaction.was_corrected = True
            corrected_interaction.correction_text = correction
            self.memory.log_interaction(corrected_interaction)

        # 4. Update performance record as incorrect
        perf = PerformanceRecord(
            interaction_number=self.interaction_count,
            was_correct=False,
            error_type=rule.domain.value,
            rules_applied_count=0,
            total_rules_available=self.memory.count_rules(),
            quality_score=0.3,
        )
        self.memory.record_performance(perf)

        # 5. Try to consolidate similar rules
        self.memory.consolidate_similar_rules()

        return rule

    def _retrieve_relevant_rules(self, user_message: str) -> list[CorrectionRule]:
        """Retrieve rules relevant to the current query."""
        all_rules = self.memory.get_all_rules(active_only=True)

        if not all_rules:
            return []

        # Build a rule summary for the LLM to match against
        rules_summary = []
        for i, rule in enumerate(all_rules):
            rules_summary.append(f"{i}: [{rule.domain.value}] {rule.rule_text}")

        rules_text = "\n".join(rules_summary)

        # Use LLM to select relevant rules
        result = llm_call(
            "You are a rule matcher. Given a user query and a numbered list of correction rules, "
            "identify which rules (by number) are relevant to the query. "
            "Return ONLY a JSON list of rule numbers, e.g. [0, 3, 5]. "
            "If no rules are relevant, return []. Be selective — only include truly relevant rules.",
            f"User query: {user_message}\n\nAvailable rules:\n{rules_text}",
            temperature=0.1,
        )

        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()
            indices = json.loads(cleaned)
            return [all_rules[i] for i in indices if isinstance(i, int) and 0 <= i < len(all_rules)]
        except (json.JSONDecodeError, TypeError, IndexError):
            # Fallback: return top-3 highest confidence rules
            return all_rules[:3] if len(all_rules) >= 3 else all_rules

    def _build_system_prompt(self, rules: list[CorrectionRule]) -> str:
        """Build a system prompt augmented with procedural rules."""
        base_prompt = (
            "You are a helpful, knowledgeable AI assistant. "
            "Answer the user's question clearly and accurately. "
            "Be concise but thorough."
        )

        if rules:
            rules_section = "\n\n--- LEARNED RULES (from past corrections) ---\n"
            rules_section += "IMPORTANT: Follow these rules carefully. They are based on past mistakes.\n\n"
            for i, rule in enumerate(rules, 1):
                rules_section += f"Rule {i} [{rule.domain.value}] (confidence: {rule.confidence:.2f}):\n"
                rules_section += f"  {rule.rule_text}\n"
                rules_section += f"  (Original mistake: {rule.original_mistake[:100]})\n\n"
            rules_section += "--- END OF RULES ---\n"
            return base_prompt + rules_section
        else:
            return base_prompt

    def _extract_rule(self, user_message: str, original_response: str, correction: str) -> CorrectionRule:
        """Extract a generalised correction rule from a specific correction."""
        result = llm_call(
            "You are a rule extraction engine. Given an original question, the agent's incorrect response, "
            "and the user's correction, extract a GENERALISED rule that will prevent this type of mistake "
            "in the future.\n\n"
            "Return a JSON object with EXACTLY these fields:\n"
            '{"rule_text": "A clear, actionable rule the agent should follow",\n'
            ' "domain": "one of: factual, formatting, tone, reasoning, accuracy, completeness, general",\n'
            ' "confidence": 0.7}\n\n'
            "The rule should be GENERAL enough to apply to similar situations, not just this specific case.\n"
            "Return ONLY valid JSON.",
            f"User question: {user_message}\n\n"
            f"Agent's incorrect response: {original_response}\n\n"
            f"User's correction: {correction}",
            temperature=0.2,
        )

        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()
            data = json.loads(cleaned)
            domain_str = data.get("domain", "general")
            try:
                domain = RuleDomain(domain_str)
            except ValueError:
                domain = RuleDomain.GENERAL

            return CorrectionRule(
                original_mistake=original_response[:500],
                correction=correction[:500],
                rule_text=data.get("rule_text", "Follow the user's correction pattern"),
                domain=domain,
                confidence=float(data.get("confidence", 0.7)),
                source_interaction_id="",
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return CorrectionRule(
                original_mistake=original_response[:500],
                correction=correction[:500],
                rule_text=f"When asked about '{user_message[:50]}', follow the correction pattern: {correction[:200]}",
                domain=RuleDomain.GENERAL,
                confidence=0.6,
            )

    def get_state(self) -> dict:
        """Get the current agent state for display."""
        total = self.memory.count_interactions()
        corrections = self.memory.count_corrections()
        rules = self.memory.count_rules()
        error_rate = self.memory.get_error_rate(window=5)

        return {
            "total_interactions": total,
            "total_corrections": corrections,
            "total_rules": rules,
            "current_error_rate": error_rate,
            "accuracy": 1.0 - error_rate if total > 0 else 0.0,
        }
