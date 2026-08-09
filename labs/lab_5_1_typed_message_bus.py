#!/usr/bin/env python3
"""
CalderR Internship – Week 5, Lab 5.1
======================================
Typed Message Bus — Structured Inter-Agent Communication

WHAT THIS LAB BUILDS:
---------------------
A message-passing backbone for multi-agent systems:
  • 4 typed Pydantic message schemas: TaskRequest, TaskResult, ErrorReport, Handoff
  • An in-memory MessageBus with topic-based routing and validation
  • 3 agents (ResearchAgent, AnalysisAgent, ReportAgent) that communicate
    exclusively through typed messages — no raw strings
  • Validation enforcement: malformed messages raise ValidationError
    before any agent receives them
  • Full message trace log for debugging and observability

WHAT THIS TEACHES YOU:
----------------------
  • Why typed inter-agent communication is the foundation of reliable
    multi-agent systems
  • How Pydantic schemas enforce message contracts at runtime
  • Topic-based publish/subscribe patterns for agent coordination
  • Message tracing for debugging complex multi-agent interactions

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────┐
    │                   MESSAGE BUS                       │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  Topics: tasks | results | errors | handoffs│    │
    │  └─────────────────────────────────────────────┘    │
    │           ▲           ▲           ▲                 │
    │           │           │           │                 │
    │    ┌──────┴──┐  ┌─────┴───┐  ┌───┴──────┐          │
    │    │Research  │  │Analysis │  │ Report   │          │
    │    │ Agent    │  │  Agent  │  │  Agent   │          │
    │    └─────────┘  └─────────┘  └──────────┘          │
    └─────────────────────────────────────────────────────┘

    Flow:
    1. ResearchAgent publishes TaskResult to "results" topic
    2. AnalysisAgent subscribes to "results", publishes analysis
    3. ReportAgent subscribes to analysis results, generates report
    4. Errors → "errors" topic, Handoffs → "handoffs" topic
    5. Malformed messages are REJECTED before delivery

Run:
    python labs/lab_5_1_typed_message_bus.py demo
    python labs/lab_5_1_typed_message_bus.py validate
    python labs/lab_5_1_typed_message_bus.py trace
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, broad-exception-caught, missing-class-docstring, missing-function-docstring, too-few-public-methods, duplicate-code
import io
import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Callable
from enum import Enum

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.rule import Rule
from rich import box

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from groq import Groq

console = Console()
app = typer.Typer(help="Lab 5.1 — Typed Message Bus for Multi-Agent Communication")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Make a single LLM call via Groq."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — PYDANTIC MESSAGE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    ERROR_REPORT = "error_report"
    HANDOFF = "handoff"


class TaskRequest(BaseModel):
    """Schema 1: A request to perform a task, sent from orchestrator to agent."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    message_type: MessageType = Field(default=MessageType.TASK_REQUEST, frozen=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender: str = Field(..., min_length=1, description="Agent ID of the sender")
    recipient: str = Field(..., min_length=1, description="Agent ID of the recipient")
    task_description: str = Field(..., min_length=5, description="What needs to be done")
    priority: Priority = Field(default=Priority.MEDIUM)
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    deadline_seconds: Optional[float] = Field(default=None, ge=1, le=3600)

    @field_validator("sender", "recipient")
    @classmethod
    def no_whitespace_names(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Agent IDs must not contain spaces")
        return v.strip()


class TaskResult(BaseModel):
    """Schema 2: The result of a completed task, sent from agent back."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    message_type: MessageType = Field(default=MessageType.TASK_RESULT, frozen=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender: str = Field(..., min_length=1)
    recipient: str = Field(..., min_length=1)
    request_id: str = Field(..., description="ID of the original TaskRequest")
    result_data: dict[str, Any] = Field(..., description="Structured result payload")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    tokens_used: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("sender", "recipient")
    @classmethod
    def no_whitespace_names(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Agent IDs must not contain spaces")
        return v.strip()


class ErrorSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ErrorReport(BaseModel):
    """Schema 3: An error report when an agent fails."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    message_type: MessageType = Field(default=MessageType.ERROR_REPORT, frozen=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender: str = Field(..., min_length=1)
    original_request_id: str = Field(..., description="ID of the failed request")
    error_code: str = Field(..., min_length=1, description="Machine-readable error code")
    error_message: str = Field(..., min_length=1, description="Human-readable description")
    severity: ErrorSeverity = Field(default=ErrorSeverity.ERROR)
    recoverable: bool = Field(default=True)
    suggested_action: Optional[str] = Field(default=None)

    @field_validator("sender")
    @classmethod
    def no_whitespace_names(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Agent IDs must not contain spaces")
        return v.strip()


class Handoff(BaseModel):
    """Schema 4: A handoff message transferring work between agents."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    message_type: MessageType = Field(default=MessageType.HANDOFF, frozen=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    from_agent: str = Field(..., min_length=1, description="Agent handing off")
    to_agent: str = Field(..., min_length=1, description="Agent receiving")
    reason: str = Field(..., min_length=5, description="Why the handoff is happening")
    context_data: dict[str, Any] = Field(default_factory=dict)
    accumulated_results: list[dict[str, Any]] = Field(default_factory=list)
    priority: Priority = Field(default=Priority.MEDIUM)

    @field_validator("from_agent", "to_agent")
    @classmethod
    def no_whitespace_names(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Agent IDs must not contain spaces")
        return v.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — IN-MEMORY MESSAGE BUS
# ═══════════════════════════════════════════════════════════════════════════

# Union of all valid message types
ValidMessage = TaskRequest | TaskResult | ErrorReport | Handoff

MESSAGE_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "tasks": TaskRequest,
    "results": TaskResult,
    "errors": ErrorReport,
    "handoffs": Handoff,
}


class MessageBus:
    """
    In-memory message bus with topic-based pub/sub and schema validation.

    Topics:
        - "tasks"    → only TaskRequest messages
        - "results"  → only TaskResult messages
        - "errors"   → only ErrorReport messages
        - "handoffs" → only Handoff messages

    Every message is validated against its topic's schema BEFORE delivery.
    Malformed messages raise ValidationError and are never delivered.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {
            "tasks": [],
            "results": [],
            "errors": [],
            "handoffs": [],
        }
        self._message_log: list[dict[str, Any]] = []
        self._rejected_count: int = 0
        self._delivered_count: int = 0

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe a callback to a topic."""
        if topic not in self._subscribers:
            raise ValueError(f"Unknown topic '{topic}'. Valid: {list(self._subscribers.keys())}")
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, message: ValidMessage) -> bool:
        """
        Publish a message to a topic. Validates schema before delivery.
        Returns True if delivered, raises ValidationError if invalid.
        """
        if topic not in MESSAGE_SCHEMA_MAP:
            raise ValueError(f"Unknown topic '{topic}'. Valid: {list(MESSAGE_SCHEMA_MAP.keys())}")

        expected_schema = MESSAGE_SCHEMA_MAP[topic]

        # Validate: message must be an instance of the expected schema
        if not isinstance(message, expected_schema):
            self._rejected_count += 1
            self._log_message(topic, message, delivered=False,
                              rejection_reason=f"Expected {expected_schema.__name__}, got {type(message).__name__}")
            raise ValidationError.from_exception_data(
                title=expected_schema.__name__,
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("message_type",),
                        "msg": f"Topic '{topic}' requires {expected_schema.__name__}, got {type(message).__name__}",
                        "input": type(message).__name__,
                        "ctx": {"error": ValueError(f"Wrong message type for topic '{topic}'")}
                    }
                ],
            )

        # Re-validate the Pydantic model (catches any corrupted fields)
        expected_schema.model_validate(message.model_dump())

        # Log and deliver
        self._log_message(topic, message, delivered=True)
        self._delivered_count += 1

        for callback in self._subscribers[topic]:
            callback(message)

        return True

    def publish_raw(self, topic: str, raw_data: dict) -> bool:
        """
        Attempt to publish raw dict data to a topic.
        The bus validates and converts to the proper schema first.
        Raises ValidationError if the data doesn't match the schema.
        """
        if topic not in MESSAGE_SCHEMA_MAP:
            raise ValueError(f"Unknown topic '{topic}'")

        expected_schema = MESSAGE_SCHEMA_MAP[topic]
        try:
            message = expected_schema.model_validate(raw_data)
        except ValidationError as e:
            self._rejected_count += 1
            self._log_message(topic, raw_data, delivered=False,
                              rejection_reason=str(e))
            raise

        return self.publish(topic, message)

    def _log_message(self, topic: str, message: Any, delivered: bool,
                     rejection_reason: str = "") -> None:
        """Log every message attempt for tracing."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "delivered": delivered,
            "message_type": type(message).__name__ if not isinstance(message, dict) else "raw_dict",
        }
        if isinstance(message, BaseModel):
            entry["message_id"] = getattr(message, "message_id", "N/A")
            entry["sender"] = getattr(message, "sender", None) or getattr(message, "from_agent", "N/A")
        if rejection_reason:
            entry["rejection_reason"] = rejection_reason
        self._message_log.append(entry)

    @property
    def trace_log(self) -> list[dict[str, Any]]:
        return list(self._message_log)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "delivered": self._delivered_count,
            "rejected": self._rejected_count,
            "total_attempts": self._delivered_count + self._rejected_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — THREE AGENTS (COMMUNICATE ONLY VIA TYPED MESSAGES)
# ═══════════════════════════════════════════════════════════════════════════

class ResearchAgent:
    """
    Agent 1: Receives a TaskRequest, performs research via LLM,
    and publishes a TaskResult to the bus.
    """
    AGENT_ID = "research-agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.processed: list[str] = []
        # Subscribe to incoming tasks
        bus.subscribe("tasks", self.handle_task)

    def handle_task(self, request: TaskRequest) -> None:
        """Handle an incoming TaskRequest."""
        if request.recipient != self.AGENT_ID:
            return  # Not for me

        console.print(f"  [cyan]{self.AGENT_ID}[/] received task: {request.task_description[:60]}...")
        start = time.time()

        # Do LLM research
        research_output = llm_call(
            system_prompt=(
                "You are a research specialist. Gather key facts and data points about "
                "the given topic. Be concise but thorough. Return 3-5 bullet points."
            ),
            user_prompt=request.task_description,
            temperature=0.6,
        )

        elapsed_ms = (time.time() - start) * 1000

        # Publish result via typed message
        result = TaskResult(
            sender=self.AGENT_ID,
            recipient="analysis-agent",
            request_id=request.message_id,
            result_data={"research_findings": research_output, "topic": request.task_description},
            confidence=0.85,
            processing_time_ms=round(elapsed_ms, 1),
        )
        self.bus.publish("results", result)
        self.processed.append(request.message_id)
        console.print(f"  [green]{self.AGENT_ID}[/] published result (confidence: {result.confidence})")


class AnalysisAgent:
    """
    Agent 2: Subscribes to results from ResearchAgent,
    analyses findings, and publishes an enhanced result + handoff.
    """
    AGENT_ID = "analysis-agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.processed: list[str] = []
        bus.subscribe("results", self.handle_result)

    def handle_result(self, result: TaskResult) -> None:
        """Handle an incoming TaskResult — analyse and enhance it."""
        if result.recipient != self.AGENT_ID:
            return

        findings = result.result_data.get("research_findings", "")
        topic = result.result_data.get("topic", "Unknown")
        console.print(f"  [yellow]{self.AGENT_ID}[/] analysing research on: {topic[:50]}...")

        start = time.time()
        analysis_output = llm_call(
            system_prompt=(
                "You are an analysis specialist. Take these research findings and: "
                "1) Identify the key insight, 2) Note any gaps or weak points, "
                "3) Provide a confidence assessment. Be concise."
            ),
            user_prompt=f"Research findings to analyse:\n{findings}",
            temperature=0.5,
        )
        elapsed_ms = (time.time() - start) * 1000

        # Publish enhanced result
        enhanced_result = TaskResult(
            sender=self.AGENT_ID,
            recipient="report-agent",
            request_id=result.request_id,
            result_data={
                "original_research": findings,
                "analysis": analysis_output,
                "topic": topic,
            },
            confidence=0.90,
            processing_time_ms=round(elapsed_ms, 1),
        )
        self.bus.publish("results", enhanced_result)

        # Also send a handoff message
        handoff = Handoff(
            from_agent=self.AGENT_ID,
            to_agent="report-agent",
            reason="Analysis complete — handing off to report generation with accumulated context",
            context_data={"topic": topic, "analysis_confidence": 0.90},
            accumulated_results=[result.result_data, {"analysis": analysis_output}],
        )
        self.bus.publish("handoffs", handoff)
        self.processed.append(result.message_id)
        console.print(f"  [green]{self.AGENT_ID}[/] published analysis + handoff")


class ReportAgent:
    """
    Agent 3: Subscribes to handoffs, generates a final structured report.
    """
    AGENT_ID = "report-agent"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.reports: list[dict] = []
        bus.subscribe("handoffs", self.handle_handoff)

    def handle_handoff(self, handoff: Handoff) -> None:
        """Handle a handoff and generate the final report."""
        if handoff.to_agent != self.AGENT_ID:
            return

        topic = handoff.context_data.get("topic", "Unknown")
        console.print(f"  [magenta]{self.AGENT_ID}[/] generating report for: {topic[:50]}...")

        accumulated = handoff.accumulated_results
        context_str = json.dumps(accumulated, indent=2, default=str)

        start = time.time()
        report_output = llm_call(
            system_prompt=(
                "You are a report specialist. Produce a concise executive summary "
                "from the research and analysis provided. Format with: "
                "TITLE, KEY FINDINGS (3 bullets), CONFIDENCE ASSESSMENT, RECOMMENDATION."
            ),
            user_prompt=f"Topic: {topic}\n\nAccumulated data:\n{context_str}",
            temperature=0.4,
        )
        elapsed_ms = (time.time() - start) * 1000

        report = {
            "topic": topic,
            "report": report_output,
            "processing_time_ms": round(elapsed_ms, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.reports.append(report)
        console.print(f"  [green]{self.AGENT_ID}[/] report generated successfully")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — DEMO, VALIDATION, AND TRACE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

def _render_trace(bus: MessageBus) -> None:
    """Render the message trace log as a rich table."""
    table = Table(
        title="Message Trace Log",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Time", style="dim", width=12)
    table.add_column("Topic", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Sender", style="yellow")
    table.add_column("Delivered", justify="center")
    table.add_column("Rejection Reason", style="red")

    for i, entry in enumerate(bus.trace_log, 1):
        ts = entry["timestamp"].split("T")[1][:8] if "T" in entry["timestamp"] else entry["timestamp"]
        delivered = "[green]YES[/]" if entry["delivered"] else "[red]NO[/]"
        rejection = entry.get("rejection_reason", "")
        if len(rejection) > 40:
            rejection = rejection[:40] + "..."
        table.add_row(
            str(i), ts, entry["topic"], entry["message_type"],
            entry.get("sender", "N/A"), delivered, rejection,
        )
    console.print(table)

    # Stats
    stats = bus.stats
    console.print(Panel(
        f"[green]Delivered:[/] {stats['delivered']}  |  "
        f"[red]Rejected:[/] {stats['rejected']}  |  "
        f"[blue]Total:[/] {stats['total_attempts']}",
        title="Bus Statistics", border_style="blue",
    ))


@app.command()
def demo():
    """Run the full 3-agent pipeline with typed message passing."""
    console.print(Rule("[bold cyan]Lab 5.1 — Typed Message Bus Demo[/]"))
    console.print()

    # Create bus and agents
    bus = MessageBus()
    _ = ResearchAgent(bus)
    _ = AnalysisAgent(bus)
    report_agent = ReportAgent(bus)

    # Show architecture
    tree = Tree("[bold]Multi-Agent Message Bus Architecture[/]")
    bus_node = tree.add("[cyan]MessageBus[/] (in-memory, topic-based)")
    bus_node.add("[dim]Topic: tasks    -> TaskRequest schema[/]")
    bus_node.add("[dim]Topic: results  -> TaskResult schema[/]")
    bus_node.add("[dim]Topic: errors   -> ErrorReport schema[/]")
    bus_node.add("[dim]Topic: handoffs -> Handoff schema[/]")
    agents_node = tree.add("[yellow]Agents[/]")
    agents_node.add("[cyan]ResearchAgent[/]  subscribes: tasks    | publishes: results")
    agents_node.add("[yellow]AnalysisAgent[/] subscribes: results  | publishes: results, handoffs")
    agents_node.add("[magenta]ReportAgent[/]   subscribes: handoffs | publishes: (final output)")
    console.print(Panel(tree, title="System Architecture", border_style="blue"))
    console.print()

    # Kickstart the pipeline with a TaskRequest
    topics = [
        "The impact of large language models on software engineering practices in 2025",
        "Emerging trends in quantum computing and their potential business applications",
    ]

    for i, topic in enumerate(topics, 1):
        console.print(Rule(f"[bold yellow]Pipeline Run {i}/{len(topics)}[/]"))
        console.print(f"[bold]Topic:[/] {topic}\n")

        request = TaskRequest(
            sender="orchestrator",
            recipient="research-agent",
            task_description=topic,
            priority=Priority.HIGH,
            context={"run_number": i},
        )

        # Publish to start the pipeline
        bus.publish("tasks", request)
        console.print()

    # Show reports
    console.print(Rule("[bold green]Generated Reports[/]"))
    for report in report_agent.reports:
        console.print(Panel(
            report["report"],
            title=f"[bold]{report['topic'][:60]}[/]",
            subtitle=f"Generated in {report['processing_time_ms']:.0f}ms",
            border_style="green",
        ))

    # Show trace
    console.print()
    _render_trace(bus)


@app.command()
def validate():
    # pylint: disable=too-many-statements
    """Demonstrate that malformed messages are rejected with ValidationError."""
    console.print(Rule("[bold cyan]Lab 5.1 — Validation Tests[/]"))
    console.print()

    bus = MessageBus()
    test_results: list[dict] = []

    # Test 1: Valid TaskRequest
    console.print("[bold]Test 1:[/] Valid TaskRequest on 'tasks' topic")
    try:
        valid_req = TaskRequest(
            sender="orchestrator",
            recipient="research-agent",
            task_description="Analyse market trends in AI",
            priority=Priority.HIGH,
        )
        bus.publish("tasks", valid_req)
        console.print("  [green]PASS[/] — Message accepted and delivered\n")
        test_results.append({"test": "Valid TaskRequest", "passed": True})
    except (ValidationError, ValueError) as e:
        console.print(f"  [red]FAIL[/] — {e}\n")
        test_results.append({"test": "Valid TaskRequest", "passed": False})

    # Test 2: Wrong schema on topic (TaskResult on "tasks" topic)
    console.print("[bold]Test 2:[/] TaskResult on 'tasks' topic (wrong schema)")
    try:
        wrong_msg = TaskResult(
            sender="agent-1",
            recipient="agent-2",
            request_id="abc123",
            result_data={"data": "test"},
            confidence=0.8,
        )
        bus.publish("tasks", wrong_msg)
        console.print("  [red]FAIL[/] — Should have been rejected!\n")
        test_results.append({"test": "Wrong schema on topic", "passed": False})
    except (ValidationError, ValueError):
        console.print("  [green]PASS[/] — Correctly rejected with ValidationError\n")
        test_results.append({"test": "Wrong schema on topic", "passed": True})

    # Test 3: Malformed raw data (missing required fields)
    console.print("[bold]Test 3:[/] Raw dict missing required fields on 'tasks' topic")
    try:
        bus.publish_raw("tasks", {"sender": "x"})  # missing recipient, task_description
        console.print("  [red]FAIL[/] — Should have been rejected!\n")
        test_results.append({"test": "Missing required fields", "passed": False})
    except ValidationError:
        console.print("  [green]PASS[/] — Correctly rejected with ValidationError\n")
        test_results.append({"test": "Missing required fields", "passed": True})

    # Test 4: Invalid confidence score (out of range)
    console.print("[bold]Test 4:[/] TaskResult with confidence > 1.0")
    try:
        _ = TaskResult(
            sender="agent-1",
            recipient="agent-2",
            request_id="abc123",
            result_data={"data": "test"},
            confidence=1.5,  # Invalid!
        )
        console.print("  [red]FAIL[/] — Should not have been created!\n")
        test_results.append({"test": "Invalid confidence", "passed": False})
    except ValidationError:
        console.print("  [green]PASS[/] — Correctly rejected at construction\n")
        test_results.append({"test": "Invalid confidence", "passed": True})

    # Test 5: Agent ID with spaces (violates validator)
    console.print("[bold]Test 5:[/] TaskRequest with spaces in agent ID")
    try:
        TaskRequest(
            sender="bad agent name",
            recipient="research-agent",
            task_description="This should fail validation",
        )
        console.print("  [red]FAIL[/] — Should not have been created!\n")
        test_results.append({"test": "Agent ID with spaces", "passed": False})
    except ValidationError:
        console.print("  [green]PASS[/] — Correctly rejected at construction\n")
        test_results.append({"test": "Agent ID with spaces", "passed": True})

    # Test 6: Empty task description (min_length=5)
    console.print("[bold]Test 6:[/] TaskRequest with too-short task description")
    try:
        TaskRequest(
            sender="orch",
            recipient="agent",
            task_description="Hi",  # Too short
        )
        console.print("  [red]FAIL[/] — Should not have been created!\n")
        test_results.append({"test": "Short task description", "passed": False})
    except ValidationError:
        console.print("  [green]PASS[/] — Correctly rejected at construction\n")
        test_results.append({"test": "Short task description", "passed": True})

    # Test 7: Invalid topic name
    console.print("[bold]Test 7:[/] Publishing to non-existent topic")
    try:
        valid_req = TaskRequest(
            sender="orch",
            recipient="agent",
            task_description="Testing invalid topic routing",
        )
        bus.publish("nonexistent", valid_req)
        console.print("  [red]FAIL[/] — Should have been rejected!\n")
        test_results.append({"test": "Invalid topic", "passed": False})
    except ValueError:
        console.print("  [green]PASS[/] — Correctly rejected with ValueError\n")
        test_results.append({"test": "Invalid topic", "passed": True})

    # Summary
    passed = sum(1 for t in test_results if t["passed"])
    total = len(test_results)
    console.print(Rule("[bold]Validation Summary[/]"))

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Test", style="bold")
    table.add_column("Result", justify="center")
    for t in test_results:
        status = "[green]PASS[/]" if t["passed"] else "[red]FAIL[/]"
        table.add_row(t["test"], status)
    console.print(table)
    console.print(f"\n[bold]{passed}/{total} tests passed[/]\n")

    # Show trace of all attempts
    _render_trace(bus)


@app.command()
def trace():
    """Run a short pipeline and display the full message trace."""
    console.print(Rule("[bold cyan]Lab 5.1 — Message Trace Mode[/]"))
    console.print()

    bus = MessageBus()
    _ = ResearchAgent(bus)
    _ = AnalysisAgent(bus)
    _ = ReportAgent(bus)

    # Single run
    request = TaskRequest(
        sender="orchestrator",
        recipient="research-agent",
        task_description="Explain the key differences between multi-agent and single-agent AI architectures",
        priority=Priority.HIGH,
    )
    bus.publish("tasks", request)

    # Also publish an error report to demonstrate that topic
    error = ErrorReport(
        sender="research-agent",
        original_request_id=request.message_id,
        error_code="RATE_LIMIT",
        error_message="API rate limit reached after 3 retries",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        suggested_action="Wait 30 seconds and retry",
    )
    bus.publish("errors", error)

    console.print()
    _render_trace(bus)

    # Show all 4 schemas
    console.print(Rule("[bold]Schema Definitions[/]"))
    schemas = [TaskRequest, TaskResult, ErrorReport, Handoff]
    for schema in schemas:
        fields_table = Table(
            title=f"[bold cyan]{schema.__name__}[/]",
            box=box.SIMPLE_HEAVY,
        )
        fields_table.add_column("Field", style="bold")
        fields_table.add_column("Type", style="yellow")
        fields_table.add_column("Required", justify="center")
        for name, field_info in schema.model_fields.items():
            req = "[green]Yes[/]" if field_info.is_required() else "[dim]No[/]"
            type_str = str(field_info.annotation).replace("typing.", "")
            if len(type_str) > 30:
                type_str = type_str[:30] + "..."
            fields_table.add_row(name, type_str, req)
        console.print(fields_table)
        console.print()


if __name__ == "__main__":
    app()
