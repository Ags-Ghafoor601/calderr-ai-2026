# Week 5 — Weekly Assessment
## Multi-Agent Systems

---

### Question 1 (Conceptual)
**Explain the difference between a supervisor pattern and a peer network. What problem does each solve best?**

A **supervisor pattern** establishes a clear hierarchy: one coordinator agent receives the overall task, decomposes it into subtasks, delegates each subtask to a specialist agent, monitors their progress, handles failures via re-routing, and aggregates the results. The supervisor is the *single point of decision-making*. Specialist agents never communicate with each other directly — all information flows through the supervisor.

A **peer network** (also called peer-to-peer or decentralised multi-agent) has no central coordinator. Every agent can communicate directly with every other agent. Agents negotiate, share partial results, and self-organise to solve the problem. There is no single point of failure, but there is also no single point of control.

| Dimension | Supervisor Pattern | Peer Network |
|-----------|-------------------|-------------|
| Coordination | Centralised — supervisor decides | Decentralised — agents negotiate |
| Failure point | Single (the supervisor) | None single, but harder to debug |
| Communication | Hub-and-spoke (through supervisor) | Mesh (any-to-any) |
| Best for | Task decomposition with clear subtasks (e.g., document processing pipeline, hiring workflow) | Open-ended exploration, brainstorming, debate where multiple perspectives must interact directly |
| Complexity | Lower — clear control flow | Higher — must handle message routing, deadlocks, convergence |
| Observability | Easier — supervisor logs all decisions | Harder — must trace across all agents |

**When each solves best:**
- **Supervisor** excels when the task has a known decomposition (e.g., "research → analyse → report"), specialists have non-overlapping domains, and you need deterministic auditability. Example: A hiring pipeline where resume scoring, bias checking, and interview question generation are distinct, sequential subtasks.
- **Peer network** excels when the problem requires agents to *challenge each other* (e.g., a debate or code review), when no single agent knows the full task structure upfront, or when resilience to individual agent failure matters more than coordination efficiency. Example: Three security reviewers independently auditing the same codebase and debating their findings.

---

### Question 2 (Conceptual)
**How does typed message passing between agents improve system reliability compared to passing raw strings or dicts?**

Typed message passing (using Pydantic models or similar) provides **compile-time-like guarantees at runtime** that raw strings and dicts cannot:

1. **Schema validation at the boundary**: Before a message is delivered to any agent, Pydantic validates every field — types, ranges, required/optional, custom constraints. A malformed message raises `ValidationError` *before* the receiving agent ever sees it. With raw dicts, an agent might receive `{"confidnece": "high"}` (misspelled key, wrong type) and silently produce garbage output.

2. **Contract enforcement**: Typed schemas act as *contracts* between agents. If the ResearchAgent promises to send a `TaskResult` with a `confidence: float` field between 0.0 and 1.0, the AnalysisAgent can rely on that guarantee. Changes to the schema break loudly (validation error) rather than silently (downstream agent misinterprets data).

3. **Discoverability and documentation**: A Pydantic model is self-documenting. `TaskRequest.model_json_schema()` produces a full JSON Schema that any developer (or any downstream agent) can inspect. Raw dicts have no inherent documentation — you must read the source code to understand the expected structure.

4. **Serialisation safety**: Pydantic handles serialisation/deserialisation consistently. A `datetime` field is always serialised the same way. With raw dicts, one agent might send `"2025-07-20"` and another `"July 20, 2025"` — the receiving agent has no guarantee.

5. **IDE support and refactoring**: Typed schemas give you autocomplete, type checking (mypy/pyright), and safe refactoring. Renaming `confidence` to `confidence_score` in the schema immediately flags every usage site. With raw dicts, renaming a key silently breaks all consumers.

**Concrete example**: In Lab 5.1, our `MessageBus` validates that only `TaskRequest` messages can be published to the "tasks" topic. If an agent accidentally sends a `TaskResult` to "tasks", the bus raises `ValidationError` immediately — rather than the downstream agent receiving an unexpected payload and producing a corrupt analysis.

---

### Question 3 (Conceptual)
**When does splitting a task across multiple agents hurt rather than help? Give two concrete failure scenarios.**

Multi-agent systems introduce coordination overhead, communication latency, and emergent failure modes. Splitting a task across agents *hurts* when:

**Failure Scenario 1: Excessive decomposition of a naturally atomic task**

Consider a task like "translate this 200-word paragraph from English to French." A single LLM call handles this excellently. If you split it across a Terminology Agent, Grammar Agent, and Style Agent — each processing the same text from a different angle — you introduce:
- **3× latency** (three LLM calls instead of one)
- **3× cost** (three sets of tokens)
- **Integration complexity**: Who reconciles when the Grammar Agent changes word order but the Style Agent prefers the original? You now need a fourth Arbitration Agent to merge, which might introduce inconsistencies that the single-agent approach never had.
- **Net result**: Worse quality, higher cost, slower execution.

**Failure Scenario 2: Agents with overlapping domains causing feedback loops**

Imagine a content creation system with a "Fact-Checker Agent" and a "Rewriting Agent." The Fact-Checker flags a claim as unverified, the Rewriting Agent softens the language, the Fact-Checker now flags the softened language as "ambiguous" and requests a rewrite, the Rewriting Agent rewrites again, and the loop continues. Without strict termination bounds:
- The system enters an infinite loop (or hits max iterations producing mediocre output)
- Each agent's correction degrades the original quality
- The final output is worse than what a single agent with "check facts and rewrite" in one prompt would produce
- **Net result**: Wasted compute, degraded quality, potential infinite loop.

**The rule of thumb**: Use multiple agents when subtasks are *genuinely independent* (different tools, different data, different expertise domains) and a single agent cannot hold all the context. If the task is naturally atomic or the agents need to pass state back and forth repeatedly, a single agent with a well-designed prompt is superior.

---

### Question 4 (Technical)
**Write the Pydantic schema for a Handoff message between a Research Agent and a Synthesis Agent. What fields are essential?**

```python
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Handoff(BaseModel):
    """
    Typed handoff message for transferring work between agents.
    Used when one agent completes its phase and passes accumulated
    context to the next agent in the pipeline.
    """
    # ── Identity ──
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this handoff"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of handoff creation"
    )

    # ── Routing (ESSENTIAL) ──
    from_agent: str = Field(
        ..., min_length=1,
        description="ID of the agent handing off work"
    )
    to_agent: str = Field(
        ..., min_length=1,
        description="ID of the agent receiving the handoff"
    )

    # ── Context (ESSENTIAL) ──
    reason: str = Field(
        ..., min_length=5,
        description="Why this handoff is happening — enables audit trail"
    )
    task_summary: str = Field(
        ..., min_length=10,
        description="Summary of the original task for the receiving agent"
    )

    # ── Accumulated State (ESSENTIAL) ──
    accumulated_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All results gathered by previous agents in the pipeline"
    )
    context_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata the receiving agent may need"
    )

    # ── Quality Signals ──
    source_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How confident the handing-off agent is in its results"
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Processing priority for the receiving agent"
    )
    requires_validation: bool = Field(
        default=False,
        description="Whether the receiving agent should validate inputs before proceeding"
    )
```

**Essential fields and why:**

| Field | Why Essential |
|-------|--------------|
| `from_agent` / `to_agent` | **Routing** — the bus/orchestrator needs to know who sends and who receives. Without this, messages cannot be delivered. |
| `reason` | **Auditability** — every handoff must explain *why* it's happening. This is the decision log that makes multi-agent systems debuggable. |
| `task_summary` | **Context transfer** — the receiving agent may not share the original context window. A summary ensures it understands the task without reading the full history. |
| `accumulated_results` | **State continuity** — multi-agent pipelines accumulate outputs. Without passing these forward, downstream agents start from scratch and the pipeline loses value. |
| `source_confidence` | **Quality signal** — the receiving agent can adjust its behaviour based on how confident the upstream agent was (e.g., lower confidence → validate more thoroughly). |
| `timestamp` + `message_id` | **Traceability** — for debugging, audit trails, and replay. Without these, you cannot reconstruct the sequence of events when something goes wrong. |

---

### Question 5 (Technical)
**How would you detect and handle a situation where two agents in a consensus system produce directly contradictory outputs with equal confidence?**

This is a fundamental challenge in multi-agent consensus. Here is a systematic approach:

**Detection:**
1. **Stance comparison**: Map each agent's output to a position on a scale (e.g., `strongly_agree` = 1.0, `strongly_disagree` = 0.0). Two agents are "directly contradictory" when their stance scores differ by more than 0.5 (e.g., one says `agree` at 0.75, the other says `disagree` at 0.25).
2. **Confidence parity check**: After detecting contradiction, check if `|confidence_A - confidence_B| < epsilon` (e.g., epsilon = 0.1). If both are within epsilon, we have a "deadlock" — neither agent can claim authority.

```python
def detect_deadlock(opinions: list[ExpertOpinion]) -> bool:
    for i, a in enumerate(opinions):
        for b in opinions[i+1:]:
            stance_diff = abs(STANCE_SCORES[a.stance] - STANCE_SCORES[b.stance])
            conf_diff = abs(a.confidence - b.confidence)
            if stance_diff > 0.5 and conf_diff < 0.1:
                return True  # Deadlock detected
    return False
```

**Handling strategies (in priority order):**

1. **Second-round deliberation with context**: Share both agents' full reasoning with each other and ask them to reconsider. Often, seeing the opposing argument causes one agent to adjust its confidence (even if not its stance), breaking the deadlock.

2. **Introduce a tie-breaking arbiter**: A fourth agent (Arbiter) — with a different model, temperature, or system prompt — reviews both arguments and picks a winner based on argument *quality*, not recency or position. The arbiter should explicitly state which argument it found more compelling and why.

3. **Surface the disagreement transparently**: If neither deliberation nor arbitration resolves the deadlock, the correct answer is to **report both positions** with a dissent log. The final output should say: "Experts disagreed — Agent A recommends X (confidence 0.7) while Agent B recommends Y (confidence 0.7). The key disagreement centres on [specific point]." Hiding the disagreement behind a forced majority vote is worse than admitting uncertainty.

4. **Escalate to human**: In production systems, equal-confidence contradiction on a critical decision should trigger a human-in-the-loop interrupt. The system preserves both arguments, the decision context, and presents them to a human reviewer.

**What NOT to do**: Never randomly pick a winner, never average contradictory positions (the average of "invest heavily" and "divest immediately" is not a useful recommendation), and never suppress the dissent.

---

### Question 6 (Design)
**You are building a multi-agent customer support system. Design the agent hierarchy: who talks to whom, what each agent knows, and what happens when the top-level agent is unavailable.**

**System: Multi-Agent Customer Support Platform**

```
                    ┌───────────────────────┐
                    │   TRIAGE AGENT (L0)   │ ← First contact, classifies intent
                    │   • Intent detection  │
                    │   • Priority scoring  │
                    │   • Route to specialist│
                    └──────────┬────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼─────────┐ ┌────▼──────────┐ ┌────▼──────────┐
   │ BILLING AGENT(L1)│ │TECHNICAL(L1)  │ │ ACCOUNT (L1)  │
   │ • Invoice lookup │ │• Error diag   │ │ • Profile mgmt│
   │ • Payment issues │ │• Config help  │ │ • Subscription│
   │ • Refund policy  │ │• Integration  │ │ • Password    │
   └────────┬─────────┘ └────┬──────────┘ └────┬──────────┘
            │                │                  │
            └──────────┬─────┘──────────────────┘
                       │
              ┌────────▼────────┐
              │ ESCALATION (L2) │ ← Complex, multi-domain, or frustrated
              │ • Full context  │
              │ • Compensation  │
              │ • Human handoff │
              └─────────────────┘
```

**Agent Roles and Knowledge Boundaries:**

| Agent | Level | Knows | Does NOT Know | Tools |
|-------|-------|-------|---------------|-------|
| **Triage Agent** | L0 | Intent taxonomy, priority rules, routing map | Domain-specific resolution details | Intent classifier, sentiment scorer |
| **Billing Agent** | L1 | Pricing plans, refund policies, invoice database schema | Technical system internals, account security | Invoice API, payment gateway, refund calculator |
| **Technical Agent** | L1 | Error codes, system architecture, known issues database | Billing details, account-level permissions | Log search, config validator, knowledge base RAG |
| **Account Agent** | L1 | Account lifecycle, subscription states, security procedures | Billing internals, system diagnostics | User database, auth system, subscription API |
| **Escalation Agent** | L2 | ALL context from previous agents, escalation playbook, compensation authority | Real-time system internals (must delegate to Technical) | All L1 tools + human handoff trigger + compensation API |

**Communication Flow:**
1. Customer query → **Triage Agent** classifies intent and priority
2. Triage sends a typed `Handoff` message to the appropriate L1 specialist
3. L1 specialist resolves OR escalates to L2 with full conversation context
4. L2 can query any L1 agent for additional information via typed `TaskRequest`
5. All responses flow back through the same path for audit logging

**Failure Handling — When the Triage Agent is unavailable:**

1. **Immediate fallback — Round-robin direct routing**: If the Triage Agent health check fails (timeout, crash), the system activates a **Fallback Router** — a lightweight, rule-based (non-LLM) component that routes based on keyword matching:
   - Keywords like "bill", "charge", "refund", "payment" → Billing Agent
   - Keywords like "error", "crash", "bug", "integration" → Technical Agent
   - Keywords like "password", "account", "cancel", "subscription" → Account Agent
   - Unmatched → Escalation Agent (with a note that triage was bypassed)

2. **Circuit breaker**: After 3 consecutive Triage Agent failures, the system switches to fallback mode permanently and alerts ops. It does not keep retrying a broken agent.

3. **Context preservation**: Even in fallback mode, all messages are logged with `triage_bypassed=True` so that when the Triage Agent recovers, the support team can audit which queries were misrouted.

4. **Graceful degradation for L1 failures**: If a specific L1 agent fails, the Triage Agent re-routes to the Escalation Agent with a note explaining which specialist was unavailable. The Escalation Agent has broader (if shallower) knowledge and can attempt resolution or trigger a human handoff.

**Key design principles:**
- **Strict knowledge boundaries**: Each agent knows only what it needs. The Billing Agent cannot access technical logs; the Technical Agent cannot issue refunds. This prevents hallucination across domains.
- **Typed messages everywhere**: Every handoff includes `customer_id`, `conversation_history`, `intent_classification`, and `priority`. No raw strings passed between agents.
- **Audit trail**: Every agent decision is logged with reasoning, enabling post-incident review and continuous improvement.
- **Human escalation as last resort, not first**: The system should resolve 80%+ of queries autonomously, escalating to humans only for genuinely complex or emotionally charged interactions.
