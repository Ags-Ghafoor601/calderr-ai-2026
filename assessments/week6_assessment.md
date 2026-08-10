# Week 6 — Weekly Assessment
## Memory Systems & Knowledge Graphs

---

### Question 1 (Conceptual)
**Explain the difference between episodic and semantic memory in an AI agent. Give a concrete example of each that matters in a production system.**

**Episodic memory** stores *specific events and interactions* — timestamped records of what happened, with whom, and in what context. Each entry is a discrete episode: "On Tuesday at 14:32, user Alice asked about quarterly revenue and I responded with Q3 2025 figures." Episodic memory preserves the full temporal and contextual detail of past interactions. It is the agent's *autobiography* — a first-person log of experiences.

**Semantic memory** stores *generalised knowledge and facts* — abstracted from specific episodes into enduring knowledge. "Alice is the CFO. She prefers concise bullet-point responses. She is primarily interested in financial metrics." Semantic memory discards the temporal context and retains only the distilled information. It is the agent's *world model* — a structured representation of what it knows.

| Dimension | Episodic Memory | Semantic Memory |
|-----------|----------------|-----------------|
| **What it stores** | Specific events with timestamps | Generalised facts and preferences |
| **Temporal context** | Preserved (when, in what session) | Discarded (timeless knowledge) |
| **Granularity** | Fine-grained (individual exchanges) | Coarse-grained (aggregate facts) |
| **Storage** | SQLite with timestamps, session IDs | ChromaDB embeddings, key-value stores |
| **Retrieval** | By recency, by session, by time range | By semantic similarity, by topic |
| **Growth** | Grows linearly with every interaction | Grows sub-linearly (facts are deduplicated) |

**Production example — Episodic:**
A customer support agent stores every interaction with each customer. When the customer calls again, the agent retrieves: "Last Tuesday you reported a billing error on invoice #4829. Our billing team resolved it and issued a $50 credit." This specific recall builds trust and avoids making the customer repeat their issue. Without episodic memory, the agent would say: "How can I help you today?" — frustrating a customer who already explained the problem.

**Production example — Semantic:**
A personal assistant that has interacted with a user over 3 months has extracted: "User is vegetarian. User prefers morning meetings. User's manager is Sarah." When the user asks the agent to schedule a lunch meeting, the semantic memory ensures the agent suggests a vegetarian-friendly restaurant and a morning time slot — without needing to retrieve the specific conversations where these preferences were mentioned.

**The critical difference**: Episodic memory answers "What happened?"; semantic memory answers "What do I know?" Both are necessary — episodic provides evidence and context, semantic provides efficient retrieval of distilled knowledge.

---

### Question 2 (Conceptual)
**What is memory consolidation? Why is it necessary, and what are the risks of getting it wrong?**

**Memory consolidation** is the process of compressing, summarising, and restructuring an agent's memory stores to keep them within manageable bounds while preserving the most important information. It is the AI equivalent of how human memory consolidates short-term memories into long-term storage during sleep.

**Why it is necessary:**

1. **Context window limits**: LLM context windows are finite (4K–128K tokens). If episodic memory grows to 10,000 entries, you cannot inject all of them into context. Consolidation compresses old episodes into summaries that fit within token budgets.

2. **Retrieval quality degradation**: As the memory store grows, retrieval accuracy drops — more candidates means more noise in similarity search. Consolidation reduces the search space to high-quality, deduplicated entries.

3. **Storage costs**: In production, storing every raw interaction for every user across months is expensive. Consolidation reduces storage by 10–50× while retaining informational value.

4. **Latency**: Querying 50,000 episodic entries is slower than querying 500 consolidated summaries. Consolidation keeps retrieval fast.

**How consolidation works in practice:**

- **Episode compression**: When episode count exceeds a threshold (e.g., 50), the oldest 25 episodes are summarised into a single "compressed memory block" using an LLM. The raw episodes are archived or deleted.
- **Importance-based retention**: High-importance memories (decisions, corrections, personal facts) are kept in full; low-importance memories (greetings, small talk) are aggressively summarised or discarded.
- **Fact extraction**: Before discarding episodes, extract any reusable facts into semantic memory. "User mentioned their birthday is March 15" → fact stored permanently, episode discarded.
- **Profile updates**: Episodic patterns are aggregated into user profile updates. Ten separate episodes mentioning Python → semantic fact: "User is proficient in Python."

**Risks of getting consolidation wrong:**

1. **Information loss**: If the summarisation model misses a critical detail (e.g., a contractual commitment mentioned once in passing), the agent permanently forgets something important. Once the raw episode is deleted, the information is irrecoverable.

2. **Hallucinated summaries**: The LLM doing consolidation might introduce errors in the summary — "User prefers Python" when they actually said "User is learning Python." This creates false semantic memories that the agent will confidently reference in future interactions.

3. **Over-aggressive forgetting**: Setting importance thresholds too high means the agent forgets most interactions, making it feel like it has no memory at all. Setting them too low defeats the purpose of consolidation.

4. **Temporal distortion**: Consolidation can merge events from different time periods into a single summary, losing the temporal ordering. The agent might confuse "User wanted X in January" with "User wanted X in July" because both were compressed into the same summary block.

5. **Bias amplification**: If consolidation preferentially retains certain types of interactions (e.g., recent over old, negative over positive), the agent develops a skewed model of the user.

---

### Question 3 (Conceptual)
**When does a knowledge graph outperform vector retrieval, and when does it fail? What types of questions expose each weakness?**

**Knowledge graphs outperform vector retrieval when:**

1. **Multi-hop reasoning**: "Who is the CEO of the company that developed GPT-4?" requires traversing: GPT-4 → developed_by → OpenAI → CEO → Sam Altman. Vector retrieval finds the paragraph about GPT-4 AND the paragraph about Sam Altman, but cannot connect them — they are separate chunks with no explicit co-occurrence of "CEO" and "GPT-4." The knowledge graph connects them through edges.

2. **Relationship queries**: "What companies has Microsoft invested in?" requires following all `invested_in` edges from the Microsoft node. Vector retrieval might find one paragraph mentioning Microsoft's investment in OpenAI, but miss the broader pattern across documents.

3. **Negation and set operations**: "Which AI companies are NOT headquartered in San Francisco?" requires enumerating all AI companies in the graph and filtering by location — trivial for a graph, impossible for vector retrieval (it cannot enumerate what it does not match).

4. **Comparison queries**: "What do Anthropic and OpenAI have in common?" requires finding shared nodes/edges between two entity subgraphs. Vector retrieval finds paragraphs about each separately but cannot compute the intersection.

**Knowledge graphs fail when:**

1. **Vague or broad queries**: "Tell me about AI safety" — no specific entities to anchor the traversal. The graph has nodes for specific safety concepts, but a broad query has no natural starting point. Vector retrieval excels here because it matches semantic similarity across the entire corpus.

2. **Incomplete extraction**: If the entity extractor missed a relationship during ingestion, the graph has a gap. "What framework powers LLaMA?" requires a `uses → PyTorch` edge. If extraction missed this, the graph cannot answer it, while the original paragraph contains the information and vector retrieval finds it.

3. **Paraphrased or implicit relationships**: "Who mentored the founders of Anthropic?" — if the graph only has `worked_at` edges (Dario Amodei → OpenAI), not `mentored_by` edges, the traversal fails. Vector retrieval might find a paragraph that implicitly discusses the mentorship relationship.

4. **Entity resolution failures**: If "Meta Platforms", "Meta", "Facebook", and "Meta AI" are stored as separate nodes instead of being merged, queries about Meta's activities are fragmented. Vector retrieval is more robust to naming variations because it operates on semantic similarity.

5. **Dense factual paragraphs**: A question about a specific number, date, or detail embedded in a paragraph ("How many parameters does LLaMA 2 have?") is a direct lookup — vector retrieval finds the exact paragraph, while graph traversal would need a `has_parameter_count` edge that might not exist.

**The GraphRAG insight**: Neither strategy is universally superior. A smart query router that classifies questions and selects the optimal retrieval strategy — or combines both — consistently outperforms either alone.

---

### Question 4 (Technical)
**Design the SQLite schema for an episodic memory store that supports recency weighting, importance scoring, and per-user isolation.**

```sql
-- ═══════════════════════════════════════════════════════
--  EPISODIC MEMORY STORE — SQLite Schema
-- ═══════════════════════════════════════════════════════

-- Core episodic memories table
CREATE TABLE episodic_memories (
    memory_id       TEXT PRIMARY KEY,           -- UUID, unique identifier
    user_id         TEXT NOT NULL,              -- Per-user isolation
    session_id      TEXT NOT NULL,              -- Groups exchanges into sessions
    timestamp       TEXT NOT NULL,              -- ISO 8601 UTC, for recency weighting
    created_epoch   REAL NOT NULL,             -- Unix epoch (seconds), for fast recency arithmetic

    -- Content
    user_message    TEXT NOT NULL,              -- What the user said
    assistant_response TEXT NOT NULL,           -- What the agent replied
    topic           TEXT DEFAULT 'general',     -- Extracted topic label

    -- Importance scoring
    importance_score REAL NOT NULL DEFAULT 0.5  -- 0.0 = trivial, 1.0 = critical
        CHECK(importance_score >= 0.0 AND importance_score <= 1.0),
    importance_reason TEXT DEFAULT '',          -- Why this score was assigned

    -- Consolidation tracking
    is_consolidated  INTEGER NOT NULL DEFAULT 0, -- 0=raw, 1=consolidated into summary
    consolidated_into TEXT DEFAULT NULL,         -- ID of the summary memory block

    -- Embedding reference (for hybrid retrieval)
    embedding_id    TEXT DEFAULT NULL            -- ChromaDB embedding ID if indexed
);

-- Per-user recency index (most common query pattern)
CREATE INDEX idx_episodic_user_recency
    ON episodic_memories(user_id, created_epoch DESC);

-- Per-session grouping
CREATE INDEX idx_episodic_session
    ON episodic_memories(session_id, created_epoch ASC);

-- High-importance memories (for consolidation decisions)
CREATE INDEX idx_episodic_importance
    ON episodic_memories(user_id, importance_score DESC);

-- Unconsolidated memories (for consolidation worker)
CREATE INDEX idx_episodic_unconsolidated
    ON episodic_memories(user_id, is_consolidated, created_epoch ASC);


-- Session summaries (compressed episodic blocks)
CREATE TABLE session_summaries (
    summary_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    summary_text    TEXT NOT NULL,              -- LLM-generated summary
    key_topics      TEXT DEFAULT '[]',          -- JSON array of topic strings
    num_exchanges   INTEGER DEFAULT 0,
    avg_importance  REAL DEFAULT 0.5
);

CREATE INDEX idx_summary_user
    ON session_summaries(user_id, timestamp DESC);


-- Consolidated memory blocks (compressed from old episodes)
CREATE TABLE consolidated_blocks (
    block_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    period_start    TEXT NOT NULL,              -- Earliest episode timestamp
    period_end      TEXT NOT NULL,              -- Latest episode timestamp
    summary_text    TEXT NOT NULL,              -- Compressed summary
    episode_count   INTEGER NOT NULL,           -- How many episodes were compressed
    key_facts       TEXT DEFAULT '[]'           -- JSON array of extracted facts
);

CREATE INDEX idx_blocks_user
    ON consolidated_blocks(user_id, created_at DESC);
```

**Design rationale:**

| Feature | Implementation | Why |
|---------|---------------|-----|
| **Recency weighting** | `created_epoch` (REAL) + descending index | Arithmetic on epoch floats is fast: `recency_score = 2^(-age_hours / half_life)`. ISO timestamp is for human readability; epoch is for computation. |
| **Importance scoring** | `importance_score` (REAL, 0-1) with CHECK constraint | Normalised to [0,1] for consistent blending with recency. CHECK constraint prevents invalid values. |
| **Per-user isolation** | `user_id` as leading column in all indexes | Every query is scoped by user. Composite index `(user_id, created_epoch DESC)` ensures SQLite never scans another user's data. |
| **Consolidation tracking** | `is_consolidated` flag + `consolidated_into` FK | The consolidation worker queries `WHERE is_consolidated = 0 ORDER BY created_epoch ASC LIMIT 25` to find the oldest unconsolidated episodes. |
| **Blended retrieval** | `embedding_id` linking to ChromaDB | Allows hybrid retrieval: recency from SQLite + relevance from ChromaDB embeddings. |

**Recency-weighted query example:**
```sql
SELECT *,
       (importance_score * 0.4 +
        (1.0 / (1.0 + (strftime('%s','now') - created_epoch) / 86400.0)) * 0.6
       ) AS blended_score
FROM episodic_memories
WHERE user_id = ?
  AND is_consolidated = 0
ORDER BY blended_score DESC
LIMIT 10;
```

---

### Question 5 (Technical)
**Explain how you would implement importance-based memory forgetting. What signals determine importance, and how do they decay over time?**

**Importance-based forgetting** is the mechanism by which an agent selectively retains high-value memories and allows low-value memories to decay, preventing unbounded memory growth while preserving the most useful information.

**Signals that determine importance:**

1. **Content-based signals (assigned at creation):**
   - **Personal information**: Names, preferences, goals, constraints → high importance (0.7–1.0)
   - **Decisions and commitments**: "Let's go with option A" → high importance (0.8–1.0)
   - **Corrections**: User correcting the agent → very high importance (0.9–1.0) because these become procedural memory
   - **Emotional content**: Frustration, praise, complaints → moderate-high (0.6–0.8)
   - **Factual queries**: "What is X?" → moderate importance (0.4–0.6)
   - **Greetings and small talk**: "Hi", "Thanks", "How are you?" → low importance (0.1–0.3)

2. **Behavioural signals (updated post-creation):**
   - **Reference count**: Every time a memory is retrieved and used in a response, its importance increases by 0.05 (capped at 1.0). Memories that keep being useful should never be forgotten.
   - **Recency of last access**: A memory last accessed 2 days ago is more important than one last accessed 6 months ago.
   - **User confirmation**: If the user confirms or builds on a past memory ("Yes, exactly, I mentioned that"), importance increases.

3. **Structural signals:**
   - **Uniqueness**: Information that appears only once (a user's birthday) is more important than information repeated often (their greeting pattern).
   - **Dependency**: If other memories reference this one (e.g., a decision that influenced subsequent conversations), it is structurally important.

**Decay function:**

The effective importance of a memory decays over time using an exponential decay model:

```python
import math
from datetime import datetime, timezone

def compute_effective_importance(
    base_importance: float,    # Original importance score (0-1)
    created_at: datetime,      # When the memory was created
    last_accessed: datetime,   # When the memory was last retrieved
    access_count: int,         # How many times it was retrieved
    half_life_days: float = 30.0,  # Importance halves every 30 days
) -> float:
    """Compute the effective importance of a memory with time decay."""
    now = datetime.now(timezone.utc)

    # Age-based decay (exponential, half-life = 30 days)
    age_days = (now - created_at).total_seconds() / 86400
    age_decay = math.pow(2, -age_days / half_life_days)

    # Access recency boost (recent access = slower decay)
    access_age_days = (now - last_accessed).total_seconds() / 86400
    access_boost = math.pow(2, -access_age_days / (half_life_days * 2))

    # Usage frequency boost (more accesses = more important)
    usage_boost = min(1.0, 0.5 + 0.1 * access_count)

    # Final effective importance
    effective = base_importance * age_decay * usage_boost + 0.2 * access_boost
    return max(0.0, min(1.0, effective))
```

**Forgetting thresholds:**
- **Effective importance < 0.15**: Memory is eligible for deletion
- **Effective importance 0.15–0.30**: Memory is eligible for consolidation (compress into summary)
- **Effective importance > 0.30**: Memory is retained in full

**Consolidation workflow:**
1. Run daily: `SELECT * FROM episodic_memories WHERE user_id = ? AND is_consolidated = 0 ORDER BY created_epoch ASC`
2. Compute effective importance for each memory
3. Memories below 0.15 → delete (after extracting any novel facts to semantic memory)
4. Memories between 0.15–0.30 → batch into groups of 10–25, summarise with LLM, store as consolidated block
5. Memories above 0.30 → keep unchanged

**Key design decisions:**
- **Never delete corrections**: Procedural memories (corrections) have a floor importance of 0.8 and never decay below 0.5 — the agent must never forget lessons learned.
- **Never delete first interactions**: The first session with any user is permanently retained — it establishes the baseline relationship.
- **Audit trail**: Deleted memories are moved to an `archived_memories` table (not truly deleted) for compliance and debugging.

---

### Question 6 (Design)
**You are building a personal AI assistant that must remember users across months of interactions without the context window growing unboundedly. Design the full memory architecture.**

## Memory Architecture: Long-Term Personal AI Assistant

### Design Principles
1. **Bounded context**: The system prompt + injected memories must never exceed 4,000 tokens
2. **Indefinite retention**: The system must handle 6+ months of daily interactions
3. **Graceful degradation**: Older memories are compressed, not deleted
4. **User control**: Users can view, edit, and delete their stored memories

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERSONAL AI ASSISTANT                         │
│                                                                 │
│  USER INPUT                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────┐     ┌─────────────────────────┐           │
│  │  WORKING MEMORY  │     │  MEMORY RETRIEVER       │           │
│  │  (current session│◄────│  ────────────────       │           │
│  │   context, last  │     │  Queries all stores,    │           │
│  │   5 exchanges)   │     │  blends recency +       │           │
│  └──────────────────┘     │  relevance + importance │           │
│                           └──────┬──────────────────┘           │
│                                  │ reads from                   │
│         ┌────────────────────────┼─────────────────────┐        │
│         │                        │                     │        │
│  ┌──────▼──────┐  ┌─────────────▼──────┐  ┌──────────▼──────┐  │
│  │ EPISODIC    │  │ SEMANTIC           │  │ PROCEDURAL      │  │
│  │ MEMORY      │  │ MEMORY             │  │ MEMORY          │  │
│  │ ──────────  │  │ ──────────         │  │ ──────────      │  │
│  │ SQLite      │  │ ChromaDB + SQLite  │  │ SQLite          │  │
│  │ Raw events  │  │ User profile,      │  │ Correction      │  │
│  │ with times  │  │ facts, preferences │  │ rules, learned  │  │
│  │ & importance│  │ embedded for       │  │ patterns        │  │
│  │             │  │ similarity search  │  │                 │  │
│  └──────┬──────┘  └─────────┬──────────┘  └─────────────────┘  │
│         │                   │                                   │
│         │    ┌──────────────┘                                   │
│         │    │                                                  │
│  ┌──────▼────▼──────────────────────────────────────────┐       │
│  │  CONSOLIDATION ENGINE (runs async, every 100 episodes)│       │
│  │  ─────────────────────────────────────────────────── │       │
│  │  1. Score effective importance (decay + usage)        │       │
│  │  2. Extract facts → semantic memory                  │       │
│  │  3. Compress old episodes → summary blocks           │       │
│  │  4. Archive low-importance memories                  │       │
│  │  5. Update user profile from patterns                │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  MEMORY INSPECTOR (Streamlit / API)                  │       │
│  │  View, edit, delete memories. See user profile.      │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Stored Where

| Memory Type | Store | What Goes In | Retention Policy |
|-------------|-------|-------------|-----------------|
| **Working** | In-memory (LangGraph state) | Current session exchanges (last 5) | Cleared on session end |
| **Episodic** | SQLite (`episodic_memories`) | Every user-assistant exchange with timestamp, importance, topic | Raw: 30 days. After 30 days: consolidated into summary blocks. Summaries: permanent. |
| **Semantic** | ChromaDB + SQLite JSON | User profile (name, preferences, goals, communication style), extracted facts | Permanent. Updated incrementally after each session. |
| **Procedural** | SQLite (`correction_rules`) | Correction rules extracted when user corrects agent. Each rule has domain, confidence, application count. | Permanent. Rules with 0 applications after 90 days are archived. |

### When Memories Are Compressed

**Trigger**: Consolidation runs when `COUNT(*) FROM episodic_memories WHERE user_id = ? AND is_consolidated = 0` exceeds 100.

**Process**:
1. Select the oldest 50 unconsolidated episodes
2. Group by session (episodes from the same session are consolidated together)
3. For each group:
   a. Extract any new facts → upsert into semantic memory
   b. Generate a 2-3 sentence summary using LLM
   c. Store summary in `consolidated_blocks` table
   d. Mark original episodes as `is_consolidated = 1`
4. Episodes with `importance_score > 0.8` are **never** consolidated — they are kept in full

### What Gets Permanently Forgotten

**Nothing is truly deleted** in the first 6 months. After 6 months:
- Consolidated summary blocks older than 6 months with `avg_importance < 0.3` are archived to cold storage (a separate SQLite file)
- User profile facts that have been contradicted by newer facts are removed (e.g., old job title replaced by new one)
- Procedural rules with 0 applications and confidence < 0.3 are deleted

**Never forgotten**:
- The user's profile (name, core preferences)
- High-importance episodic memories (decisions, commitments)
- All procedural corrections (lessons learned)
- The first session with the user (establishes baseline)

### Context Window Budget (4,000 tokens max)

| Component | Token Budget | Content |
|-----------|-------------|---------|
| System prompt | ~300 tokens | Base personality, instructions |
| User profile | ~200 tokens | Structured profile summary |
| Retrieved memories (top-5) | ~1,500 tokens | 5 memories × ~300 tokens each |
| Active procedural rules | ~500 tokens | Top 5 relevant correction rules |
| Current session context | ~1,000 tokens | Last 3-5 exchanges |
| Safety margin | ~500 tokens | Buffer for LLM response |

This budget ensures the context window never exceeds model limits, even after months of interaction.
