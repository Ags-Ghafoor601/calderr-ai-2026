# Week 4 — Weekly Assessment
## LangGraph, Workflows & State Machines

---

### Question 1 (Conceptual)
**What does a graph-based architecture give you that a linear chain cannot? Give a concrete example.**

A **linear chain** processes steps sequentially — A → B → C → D — with no ability to deviate, loop back, or skip steps based on intermediate results. A **graph-based architecture** (like LangGraph's StateGraph) provides three critical capabilities that chains cannot:

1. **Conditional Branching**: A graph can route execution to different nodes based on runtime state. For example, in a content moderation system, after classifying content the graph can branch three ways:
   - Safe content → auto-approve path
   - Borderline content → human-review path (with interrupt)
   - Harmful content → auto-reject path

   A linear chain would have to process every step for every input, wasting resources and making the flow unnecessarily rigid.

2. **Cyclic Loops**: Graphs support cycles — a node can route back to a previous node. For example, a self-correcting agent generates a response, validates it, and if validation fails, loops back to regenerate with feedback. This continues up to N iterations. A linear chain has no concept of "go back to step 2."

3. **Parallel Execution**: Graphs can fan out to multiple nodes simultaneously (e.g., researching three topics in parallel) and then fan in to a synthesis node — something inherently impossible in a single linear chain.

**Concrete example**: Consider a document processing pipeline. In a linear chain, every document goes through the same fixed steps regardless of size. With a graph, after the validation node, a conditional edge checks `is_oversized`:
- If `True` → route to a `split_document` node first, then chunk
- If `False` → skip directly to `chunk_document`

This conditional routing, which is natural in a graph, would require awkward workarounds in a linear chain (e.g., wrapping everything in if/else blocks that break the chain abstraction).

---

### Question 2 (Conceptual)
**Explain the role of a checkpointer in LangGraph. Why is persistence important for human-in-the-loop workflows?**

A **checkpointer** in LangGraph is a persistence mechanism that saves the complete graph state (all TypedDict fields) at each step of execution. LangGraph provides several implementations:

| Checkpointer | Backend | Use Case |
|-------------|---------|----------|
| `MemorySaver` | In-memory dict | Development/testing only |
| `SqliteSaver` | SQLite database | Local persistence, survives process restarts |
| `PostgresSaver` | PostgreSQL | Production multi-server deployments |

**How it works**:
1. Before each node executes, the checkpointer serialises the current state and saves it keyed by `(thread_id, step_number)`
2. If the graph is interrupted or the process crashes, the state is preserved
3. When the graph resumes, it loads the latest checkpoint and continues from the exact point of interruption

**Why persistence is critical for human-in-the-loop (HITL)**:

In a HITL workflow, the graph pauses execution using `interrupt_before` or `interrupt_after` to wait for human input. This wait could last minutes, hours, or even days. Without persistence:
- The graph state lives only in memory → process restart = lost state
- The human reviewer would need to re-trigger the entire workflow from scratch
- Multiple concurrent workflows couldn't be tracked independently

With a checkpointer like `SqliteSaver`:
- The state is durably stored, surviving process restarts
- A human can review the pending item hours later, and the graph resumes from the exact checkpoint
- Multiple threads (identified by `thread_id`) run independently
- The full state history is preserved for auditing and debugging

In my Lab 4.3 implementation, I used `SqliteSaver` to persist content moderation state. When borderline content reaches the `human_review` node, the graph interrupts before `apply_human_decision`. The moderator can review later, call `update_state` with their decision, and then `invoke(None, config)` to resume — the graph picks up exactly where it left off.

---

### Question 3 (Conceptual)
**What is the difference between a conditional edge and a regular edge? When would you use each?**

| Aspect | Regular Edge | Conditional Edge |
|--------|-------------|-----------------|
| **Definition** | A fixed, unconditional connection from node A to node B | A dynamic connection where a routing function decides which node to go to next |
| **Syntax** | `graph.add_edge("A", "B")` | `graph.add_conditional_edges("A", route_fn, {"option1": "B", "option2": "C"})` |
| **Determinism** | Always takes the same path | Path depends on current state at runtime |
| **Use case** | Sequential steps that always follow each other | Decision points where the next step depends on intermediate results |

**When to use each**:

**Regular edges** are appropriate when:
- One step always follows another regardless of state (e.g., `load_document` → `validate_document`)
- You're building the "backbone" of a pipeline where steps have a fixed order
- After a conditional branch converges, all paths lead to the same next step (e.g., both `auto_approve` and `auto_reject` → `log_decision`)

**Conditional edges** are appropriate when:
- The next step depends on runtime data (e.g., document size, validation result, classification)
- You need to implement branching logic (if/else or switch/case patterns)
- You need to implement loops (the routing function can send execution back to an earlier node)
- You need to terminate early (routing to `END` on certain conditions)

**Example from my implementation**:
```python
# Regular edge — validation always follows loading
graph.add_edge("load_document", "validate_document")

# Conditional edge — routing depends on validation result AND document size
graph.add_conditional_edges(
    "validate_document",
    route_after_validation,     # This function reads state and returns a string
    {
        "handle_invalid": "handle_invalid",
        "split_document": "split_document",
        "chunk_document": "chunk_document",
    },
)
```

---

### Question 4 (Technical)
**Write the TypedDict state schema for a graph that needs to accumulate a list of messages and a single draft string.**

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """State schema for a graph that accumulates messages and maintains a draft.

    - messages: Uses Annotated[..., add] so each node's returned messages
      are APPENDED to the existing list (reducer pattern), rather than
      replacing the entire list. This is critical for multi-turn conversations
      where you want to preserve full message history.

    - draft: A plain string that gets REPLACED on each update. This is
      intentional — each revision of the draft supersedes the previous one.
    """

    # Accumulates across nodes — each node's return is appended, not replaced.
    # The `add` operator (from operator module) acts as the reducer:
    # new_state["messages"] = old_state["messages"] + node_return["messages"]
    messages: Annotated[list[BaseMessage], add]

    # Replaced on each update — stores the latest working draft.
    # No reducer annotation means standard replacement semantics:
    # new_state["draft"] = node_return["draft"]
    draft: str
```

**Why this design matters**:

The key insight is the **reducer annotation** on `messages`. Without `Annotated[list[BaseMessage], add]`, if a node returns `{"messages": [new_msg]}`, it would **replace** the entire message list. With the `add` reducer, it **appends** the new messages to the existing list.

For `draft`, we intentionally do NOT use a reducer because each draft revision should replace the previous one — we only care about the latest version.

---

### Question 5 (Technical)
**Explain how an interrupt pauses a LangGraph execution and how the graph later resumes from that exact point.**

LangGraph's interrupt mechanism works through the checkpointer system:

**Pausing (Interrupt)**:

1. When building the graph, you specify interrupt points:
   ```python
   graph.compile(
       checkpointer=SqliteSaver.from_conn_string("state.db"),
       interrupt_before=["apply_human_decision"]  # Pause BEFORE this node
   )
   ```

2. During execution, the graph processes nodes normally. When it reaches a node listed in `interrupt_before` (or after a node in `interrupt_after`):
   - The current complete state is **serialised and saved** to the checkpointer database, keyed by `(thread_id, checkpoint_id)`
   - The graph **stops execution** and returns the current state to the caller
   - The returned state includes `state.next` indicating which node would execute next

3. The caller receives the partial result and can inspect it — showing the human reviewer what needs attention.

**Resuming**:

1. The human (or external system) provides their decision by updating the state:
   ```python
   config = {"configurable": {"thread_id": "abc123"}}
   graph.update_state(config, {
       "human_decision": "approve",
       "human_notes": "Content is acceptable"
   })
   ```

2. The graph resumes by calling `invoke` with `None` as the input and the same config:
   ```python
   result = graph.invoke(None, config)
   ```

3. Internally, LangGraph:
   - Loads the latest checkpoint for `thread_id = "abc123"` from the database
   - Merges the updated fields into the saved state
   - Resumes execution from the interrupted node (`apply_human_decision`)
   - Continues processing subsequent nodes normally until completion or another interrupt

**Key property**: The resume is exact — it starts from the precise node where execution paused, with all accumulated state preserved (including processing logs, classification results, etc.). This works even if the process was restarted between interrupt and resume, because the state is persisted in SQLite/PostgreSQL, not in memory.

---

### Question 6 (Design)
**Design a LangGraph workflow for a content moderation system with both an automatic path and a human-review path.**

```
                         ┌──────────────┐
                         │ receive_post │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │classify_post │
                         │ (LLM + rules)│
                         └──────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
          ┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
          │ auto_approve│ │  human   │ │ auto_reject│
          │  (safe)     │ │  review  │ │ (harmful)  │
          └──────┬──────┘ │(bordline)│ └─────┬──────┘
                 │        └────┬─────┘       │
                 │          ⏸️ INTERRUPT       │
                 │        ┌────▼──────┐      │
                 │        │  apply    │      │
                 │        │ decision  │      │
                 │        └────┬──────┘      │
                 │             │              │
                 └─────────────┼──────────────┘
                        ┌──────▼───────┐
                        │ log_decision │
                        │  (audit)     │
                        └──────┬───────┘
                               │
                              END
```

**State Schema**:
```python
class ModerationState(TypedDict):
    post_content: str                          # The content to moderate
    post_id: str                               # Unique identifier
    classification: str                        # safe / borderline / harmful
    confidence: float                          # Classifier confidence (0-1)
    classification_reason: str                 # Why classified this way
    human_decision: str                        # approve / reject (human input)
    human_notes: str                           # Optional reviewer notes
    final_decision: str                        # Final outcome
    decision_source: str                       # "auto" or "human"
    processing_log: Annotated[list[str], add]  # Accumulated audit trail
```

**Design Decisions**:

1. **Three-way routing** (not just binary) — safe content doesn't waste human time, harmful content is blocked immediately, only ambiguous content needs review
2. **Interrupt before `apply_decision`** — not after `human_review` — so the reviewer can see the classification but the graph doesn't proceed until they decide
3. **SqliteSaver persistence** — allows reviewers to handle queues asynchronously, even across process restarts
4. **Audit logging node** — every decision (auto or human) is logged with timestamp, classification, and source for compliance
5. **Annotated processing_log** — uses `add` reducer so every node's logs accumulate automatically without manual list management
