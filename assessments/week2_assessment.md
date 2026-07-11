# Week 2 Assessment — Prompt Engineering & Tool Calling

## 1. [Conceptual] What is the difference between Chain-of-Thought and Tree-of-Thought prompting? When would you use each?

**Chain-of-Thought (CoT)** prompting encourages the LLM to reason through a problem step-by-step in a *single linear sequence* before arriving at a final answer. It works by including phrases like "Let's think step by step" or by providing few-shot examples that demonstrate intermediate reasoning steps. CoT is deterministic in the sense that the model follows one reasoning path from start to finish.

**Tree-of-Thought (ToT)** prompting, introduced by Yao et al. (2023), extends CoT by allowing the model to explore *multiple reasoning paths simultaneously*, like a tree of possibilities. At each step, the model generates several candidate "thoughts," evaluates which ones are most promising (using the LLM itself as a heuristic evaluator), and then explores the best branches further. It can also backtrack if a path leads to a dead end.

**When to use each:**
- **Use CoT** for straightforward multi-step problems where the reasoning path is mostly linear — arithmetic word problems, logical deductions, following a recipe of instructions, or explaining a concept. It adds minimal latency and cost.
- **Use ToT** for complex problems where there is genuine ambiguity about the correct approach and the model might go down a wrong path — creative writing with constraints, puzzle-solving (e.g., Game of 24, crosswords), planning tasks, or problems where exploration and backtracking are essential. ToT is more expensive (multiple LLM calls) but significantly more capable on hard reasoning tasks.

---

## 2. [Conceptual] Why does structured output matter for production AI systems? What problems does it solve?

Structured output matters because **production systems need predictable, parseable, and validated data** — not free-form text. When an LLM returns a plain string, downstream code has to perform brittle regex parsing or string matching to extract the information it needs, which breaks whenever the model phrases things slightly differently.

**Problems structured output solves:**

1. **Reliability**: By forcing the LLM to output valid JSON matching a Pydantic schema, you eliminate entire classes of parsing errors. The output is *guaranteed* to have the right fields, types, and structure.

2. **Type Safety**: Pydantic validates that a salary is an integer (not the string "competitive"), that a date is actually a date, and that an enum field contains only allowed values. This prevents invalid data from propagating through your system.

3. **Integration**: Structured output plugs directly into databases, APIs, and UI components. A validated `JobPosting` object can be inserted into SQL, serialized to JSON for a REST endpoint, or rendered in a frontend — with zero manual transformation.

4. **Testability**: When outputs have a fixed schema, you can write automated tests that assert specific fields exist, values are within expected ranges, and required relationships hold. Free-text outputs are nearly impossible to test reliably.

5. **Composability**: Structured outputs from one LLM call can be passed as structured *inputs* to the next call in a chain or pipeline, enabling multi-step agent architectures where each step builds on the previous one's validated output.

---

## 3. [Conceptual] Explain the tool-calling lifecycle: how does a model decide to call a tool, and what happens after?

The tool-calling lifecycle in a modern LLM agent follows these steps:

**1. Schema Binding**: Before any user query, tool schemas (function name, description, parameter types) are attached to the model via `bind_tools()`. These schemas are injected into the system prompt so the model knows what tools exist and how to call them.

**2. Decision**: When a user query arrives, the model examines the query and its available tool schemas. Based on the tool descriptions and the query's intent, the model decides whether to:
   - Respond directly (no tool needed), or
   - Generate one or more **tool call requests** — structured JSON objects containing the tool name and arguments.

**3. Tool Call Generation**: The model outputs a special response (an `AIMessage` with `tool_calls` populated) instead of regular text. Each tool call contains:
   - `name`: which tool to invoke (e.g., `"calculate"`)
   - `args`: the arguments as a JSON object (e.g., `{"expression": "sqrt(144)"}`)
   - `id`: a unique identifier to match the result back to the call

**4. Execution**: The application code (not the model) intercepts these tool calls, routes them to the actual Python functions, executes them, and collects the results.

**5. Result Injection**: The tool results are wrapped in `ToolMessage` objects (containing the result string and the matching `tool_call_id`) and appended to the conversation history.

**6. Synthesis**: The model is called again with the updated history (including tool results). It now has the real data from the tools and generates a natural language response that incorporates those results.

**7. Iteration** (optional): If the model determines it needs more information, it can issue additional tool calls, creating a loop (the ReAct pattern) until it has enough information to provide a final answer.

---

## 4. [Technical] Write a Pydantic model for extracting a job posting's title, salary range, and required skills.

```python
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class JobPosting(BaseModel):
    """Structured representation of a job posting extracted from unstructured text."""

    title: str = Field(
        description="The job title, e.g. 'Senior Python Developer'"
    )
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum annual salary in USD. None if not mentioned."
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum annual salary in USD. None if not mentioned."
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="List of required technical skills, e.g. ['Python', 'FastAPI', 'SQL']"
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Job title must not be empty")
        return v.strip()

    @field_validator("required_skills")
    @classmethod
    def deduplicate_skills(cls, v: list[str]) -> list[str]:
        seen = set()
        result = []
        for skill in v:
            s = skill.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)
        return result

    @model_validator(mode="after")
    def validate_salary_range(self) -> "JobPosting":
        if (self.salary_min is not None and self.salary_max is not None
                and self.salary_min > self.salary_max):
            self.salary_min, self.salary_max = self.salary_max, self.salary_min
        return self
```

Key design decisions:
- `salary_min` and `salary_max` are `Optional[int]` because many postings don't disclose salary.
- `required_skills` is a `list[str]` with a deduplication validator to prevent the LLM from listing the same skill twice.
- The `model_validator` auto-corrects swapped salary values instead of raising an error, making extraction more robust.
- `Field(description=...)` provides context to the LLM when using `with_structured_output()`.

---

## 5. [Technical] What is exponential backoff and why is it used when calling external APIs?

**Exponential backoff** is a retry strategy where the wait time between consecutive retry attempts increases exponentially. The formula is typically:

```
wait_time = base_delay × 2^attempt + random_jitter
```

For example, with a base delay of 1 second:
- Attempt 1: wait ~1s
- Attempt 2: wait ~2s
- Attempt 3: wait ~4s
- Attempt 4: wait ~8s
- Attempt 5: wait ~16s (often capped at a maximum)

**Why it's used:**

1. **Prevents Thundering Herd**: If 1,000 clients all hit a rate limit at the same time and all retry after exactly 1 second, they'll *all* hit the API again simultaneously, causing the same overload. Random jitter spreads retries over time.

2. **Respects Server Recovery**: When an API returns a 429 (rate limit) or 503 (service unavailable), the server needs time to recover. Exponentially increasing delays give the server progressively more breathing room.

3. **Avoids Wasted Requests**: A fixed short delay (e.g., always retry after 1s) wastes API quota on retries that are almost certain to fail if the rate limit window hasn't reset. Longer waits mean each retry has a higher probability of success.

4. **Cost Efficiency**: Each failed API call consumes resources (network, compute, API credits). Fewer, better-spaced retries reduce total cost while maintaining a high probability of eventual success.

5. **Industry Standard**: Major cloud providers (AWS, Google Cloud, Azure) all recommend exponential backoff in their API documentation. Libraries like `tenacity` (Python) and `axios-retry` (JS) implement it by default.

---

## 6. [Design] Design a tool schema for a 'send_email' function. What parameters, types, and descriptions would you include?

```json
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to one or more recipients. Supports plain text and HTML body, CC/BCC, and attachments. Use this when the user explicitly asks to send, compose, or forward an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "List of primary recipient email addresses"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line (max 200 characters)"
                },
                "body": {
                    "type": "string",
                    "description": "The email body content in plain text"
                },
                "body_html": {
                    "type": "string",
                    "description": "Optional HTML version of the email body. If provided, recipients with HTML-capable clients will see this instead of the plain text body."
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "Optional list of CC (carbon copy) recipient email addresses"
                },
                "bcc": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "Optional list of BCC (blind carbon copy) recipient email addresses"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "Email priority level. Defaults to 'normal'."
                },
                "reply_to": {
                    "type": "string",
                    "format": "email",
                    "description": "Optional reply-to address if different from sender"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
}
```

**Design rationale:**
- **`to` as an array**: Supports multiple recipients natively, which is more common than single-recipient emails in business contexts.
- **`body` + `body_html`**: Separating plain text and HTML ensures the email works for all clients (plain text as fallback, HTML for rich formatting).
- **`cc`/`bcc` as optional arrays**: Standard email features that the agent should support but not require.
- **`priority` as an enum**: Constrained to 3 valid values, preventing the LLM from generating invalid priorities.
- **`required` is minimal**: Only `to`, `subject`, and `body` are required — everything else has sensible defaults or is optional.
- **Descriptions are specific**: Each parameter description tells the LLM exactly what format and content is expected, reducing hallucination.
