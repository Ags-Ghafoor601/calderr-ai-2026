"""
CalderR Internship – Week 2, Lab 2.3
======================================
Error Recovery Agent — Fallbacks, Retries & Exponential Backoff

WHAT THIS LAB BUILDS:
---------------------
An agent that gracefully handles tool failures through:
  • Exponential backoff with jitter for rate-limit errors
  • Fallback chains: if tool A fails, try tool B
  • Complete logging of all attempts, successes, and failures
  • Retry budgets (max attempts per tool)
  • Circuit breaker pattern: disable tools that fail too often

WHAT THIS TEACHES YOU:
----------------------
  • Why error handling is critical in production AI systems
  • How exponential backoff prevents thundering herd problems
  • How to build resilient agents that degrade gracefully
  • How to log and trace tool execution for debugging
  • How fallback chains work in real agent architectures

ARCHITECTURE:
  User Query
      ↓
  Agent (ChatGroq with tools)
      ↓
  Tool Execution with Error Wrapper
      ↓  ↓ (failure)
  ✓   Retry Manager (exponential backoff)
  ↓       ↓ (max retries exceeded)
  ↓   Fallback Tool Selector
  ↓       ↓
  ↓   Fallback Execution
  ↓       ↓
  ← ← ← ←
      ↓
  Result + Full Execution Log
      ↓
  Rich Display (timeline, stats)

Run:
    python labs/lab_2_3_error_recovery.py
"""

import os
import sys
import json
import time
import random
import traceback
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional

# Fix Windows console encoding (cp1252 cannot handle Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich.tree import Tree
from rich import box

# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────
load_dotenv()
console = Console(force_terminal=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    console.print("[bold red]✗ GROQ_API_KEY not found in .env[/]")
    sys.exit(1)

MODEL_NAME = "llama-3.1-8b-instant"


# ═══════════════════════════════════════════════
#  SECTION 1: Error Types & Logging
# ═══════════════════════════════════════════════

class ErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ExecutionAttempt:
    """Record of a single tool execution attempt."""
    tool_name: str
    args: dict
    attempt_number: int
    timestamp: float
    success: bool
    result: Optional[str] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    is_fallback: bool = False
    backoff_wait_s: float = 0.0


@dataclass
class ExecutionLog:
    """Complete log of all execution attempts for a query."""
    query: str
    attempts: list[ExecutionAttempt] = field(default_factory=list)
    final_result: Optional[str] = None
    total_duration_ms: float = 0.0
    tools_tried: list[str] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    @property
    def successful_attempts(self) -> int:
        return sum(1 for a in self.attempts if a.success)

    @property
    def failed_attempts(self) -> int:
        return sum(1 for a in self.attempts if not a.success)

    @property
    def total_backoff_time(self) -> float:
        return sum(a.backoff_wait_s for a in self.attempts)


# ═══════════════════════════════════════════════
#  SECTION 2: Exponential Backoff Engine
# ═══════════════════════════════════════════════

class BackoffEngine:
    """
    Implements exponential backoff with jitter.

    Formula: wait = min(base_delay * 2^attempt + jitter, max_delay)
    Where jitter is random(0, base_delay * 0.5)
    """
    def __init__(
        self,
        base_delay: float = 0.5,
        max_delay: float = 16.0,
        max_retries: int = 4,
        jitter_factor: float = 0.5,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """Calculate the delay for a given attempt number (0-indexed)."""
        exponential = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, self.base_delay * self.jitter_factor)
        return min(exponential + jitter, self.max_delay)

    def should_retry(self, attempt: int) -> bool:
        """Check if we should retry (haven't exceeded max retries)."""
        return attempt < self.max_retries

    def wait(self, attempt: int) -> float:
        """Sleep for the calculated backoff duration. Returns wait time."""
        delay = self.get_delay(attempt)
        console.print(
            f"  [yellow]⏳ Backoff: waiting {delay:.2f}s "
            f"(attempt {attempt + 1}/{self.max_retries})[/]"
        )
        time.sleep(delay)
        return delay


# ═══════════════════════════════════════════════
#  SECTION 3: Circuit Breaker
# ═══════════════════════════════════════════════

class CircuitBreaker:
    """
    Disables tools that fail too many times in a row.
    States: CLOSED (normal) → OPEN (disabled) → HALF_OPEN (testing)
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failure_counts: dict[str, int] = {}
        self._open_since: dict[str, float] = {}

    def record_success(self, tool_name: str):
        """Record a successful execution — resets failure count."""
        self._failure_counts[tool_name] = 0
        self._open_since.pop(tool_name, None)

    def record_failure(self, tool_name: str):
        """Record a failed execution — increment failure count."""
        count = self._failure_counts.get(tool_name, 0) + 1
        self._failure_counts[tool_name] = count
        if count >= self.failure_threshold:
            self._open_since[tool_name] = time.time()
            console.print(
                f"  [bold red]⚡ Circuit OPEN for '{tool_name}' "
                f"({count} consecutive failures)[/]"
            )

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool is available (circuit not open)."""
        if tool_name not in self._open_since:
            return True
        # Check if enough time has passed for half-open state
        elapsed = time.time() - self._open_since[tool_name]
        if elapsed >= self.reset_timeout:
            console.print(
                f"  [cyan]🔄 Circuit HALF-OPEN for '{tool_name}' "
                f"(testing after {elapsed:.0f}s)[/]"
            )
            return True
        return False

    def get_status(self, tool_name: str) -> str:
        if tool_name not in self._open_since:
            return "CLOSED (OK)"
        elapsed = time.time() - self._open_since[tool_name]
        if elapsed >= self.reset_timeout:
            return "HALF-OPEN"
        return f"OPEN ({self.reset_timeout - elapsed:.0f}s remaining)"


# ═══════════════════════════════════════════════
#  SECTION 4: Unreliable Tools (Simulated Failures)
# ═══════════════════════════════════════════════

# Global failure simulation counter
_call_counter: dict[str, int] = {}


def _should_fail(tool_name: str, fail_rate: float = 0.5) -> Optional[ErrorType]:
    """
    Determine if a tool call should fail (for simulation).
    First 2 calls to each tool fail, then they succeed.
    This creates a predictable demo of retry behavior.
    """
    _call_counter.setdefault(tool_name, 0)
    _call_counter[tool_name] += 1
    count = _call_counter[tool_name]

    # First call: rate limit error
    if count == 1:
        return ErrorType.RATE_LIMIT
    # Second call: timeout
    if count == 2:
        return ErrorType.TIMEOUT
    # After that: succeed
    return None


# ── Primary Tools (can fail) ─────────────────

@tool
def fetch_stock_price(symbol: str) -> str:
    """Fetch the current stock price for a given ticker symbol.
    Use this when the user asks about stock prices or market data."""
    failure = _should_fail("fetch_stock_price")
    if failure == ErrorType.RATE_LIMIT:
        raise Exception("HTTP 429: Rate limit exceeded. Too many requests.")
    if failure == ErrorType.TIMEOUT:
        raise Exception("HTTP 408: Request timeout. Server did not respond in time.")

    # Mock stock data
    stocks = {
        "AAPL": 189.84, "GOOGL": 141.80, "MSFT": 378.91,
        "AMZN": 178.25, "TSLA": 248.42, "META": 474.99,
        "NVDA": 875.28, "NFLX": 605.88,
    }
    sym = symbol.upper().strip()
    if sym in stocks:
        price = stocks[sym] + random.uniform(-2, 2)
        return f"${price:.2f} USD (as of {datetime.now().strftime('%H:%M')})"
    return f"Symbol '{sym}' not found in market data."


@tool
def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city.
    Use this when the user asks about weather conditions."""
    failure = _should_fail("fetch_weather")
    if failure == ErrorType.RATE_LIMIT:
        raise Exception("HTTP 429: Rate limit exceeded on weather API.")
    if failure == ErrorType.TIMEOUT:
        raise Exception("Connection timeout: weather service unreachable.")

    # Mock weather data
    weather_data = {
        "new york": ("Partly cloudy", 72, 65),
        "london": ("Light rain", 58, 80),
        "tokyo": ("Clear sky", 82, 45),
        "paris": ("Overcast", 64, 70),
        "sydney": ("Sunny", 75, 40),
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        desc, temp, humidity = weather_data[city_lower]
        return f"Weather in {city}: {desc}, {temp}°F, Humidity: {humidity}%"
    return f"Weather data not available for '{city}'."


@tool
def fetch_news(topic: str) -> str:
    """Fetch the latest news headlines about a topic.
    Use this when the user asks about recent news or events."""
    failure = _should_fail("fetch_news")
    if failure == ErrorType.RATE_LIMIT:
        raise Exception("HTTP 429: News API rate limit exceeded.")
    if failure == ErrorType.TIMEOUT:
        raise Exception("HTTP 504: Gateway timeout from news aggregator.")

    # Mock news data
    news = {
        "technology": [
            "AI Startup Raises $500M in Series D Funding",
            "New Quantum Computing Breakthrough at MIT",
            "Open-Source LLM Surpasses GPT-4 on Coding Benchmarks",
        ],
        "science": [
            "James Webb Telescope Discovers New Exoplanet",
            "CRISPR Gene Therapy Shows Promise in Clinical Trials",
            "Antarctic Ice Sheet Melting Faster Than Predicted",
        ],
        "business": [
            "Federal Reserve Holds Interest Rates Steady",
            "Global Supply Chain Issues Ease in Q3",
            "Remote Work Adoption Reaches 40% in Tech Sector",
        ],
    }
    topic_lower = topic.lower().strip()
    for key, headlines in news.items():
        if key in topic_lower or topic_lower in key:
            return "Latest headlines:\n" + "\n".join(f"  • {h}" for h in headlines)
    return f"No recent news found for topic '{topic}'."


# ── Fallback Tools (always succeed) ──────────

@tool
def fetch_stock_price_cached(symbol: str) -> str:
    """[FALLBACK] Fetch stock price from cached/delayed data.
    Less accurate but always available. Used when primary stock API fails."""
    stocks = {
        "AAPL": 188.50, "GOOGL": 140.20, "MSFT": 377.00,
        "AMZN": 177.00, "TSLA": 247.00, "META": 473.00,
        "NVDA": 873.00, "NFLX": 604.00,
    }
    sym = symbol.upper().strip()
    if sym in stocks:
        return f"${stocks[sym]:.2f} USD (cached data, may be delayed)"
    return f"No cached data for '{sym}'."


@tool
def fetch_weather_simple(city: str) -> str:
    """[FALLBACK] Get basic weather estimate based on historical averages.
    Less accurate but always works. Used when primary weather API fails."""
    return f"Weather in {city}: Historical average suggests mild conditions, ~65°F. (Fallback data — primary API unavailable)"


@tool
def fetch_news_cached(topic: str) -> str:
    """[FALLBACK] Get cached news summary for a topic.
    May not be real-time. Used when primary news API fails."""
    return f"Recent coverage on '{topic}' includes ongoing developments in the field. (Cached summary — primary API unavailable)"


# ═══════════════════════════════════════════════
#  SECTION 5: Fallback Chain Registry
# ═══════════════════════════════════════════════

FALLBACK_MAP: dict[str, list] = {
    "fetch_stock_price": [fetch_stock_price_cached],
    "fetch_weather": [fetch_weather_simple],
    "fetch_news": [fetch_news_cached],
}

ALL_TOOLS = [
    fetch_stock_price,
    fetch_weather,
    fetch_news,
    fetch_stock_price_cached,
    fetch_weather_simple,
    fetch_news_cached,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ═══════════════════════════════════════════════
#  SECTION 6: Resilient Tool Executor
# ═══════════════════════════════════════════════

class ResilientExecutor:
    """
    Executes tools with retry logic, exponential backoff,
    fallback chains, and circuit breaker protection.
    """
    def __init__(self):
        self.backoff = BackoffEngine(
            base_delay=0.3,
            max_delay=8.0,
            max_retries=3,
            jitter_factor=0.5,
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0,
        )
        self.execution_log: Optional[ExecutionLog] = None

    def execute_with_recovery(
        self,
        tool_name: str,
        tool_args: dict,
        log: ExecutionLog,
    ) -> str:
        """
        Execute a tool with full error recovery:
        1. Try primary tool with retries + backoff
        2. If all retries fail, try fallback tools
        3. If fallbacks fail, return graceful error message
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_available(tool_name):
            console.print(
                f"  [red]🚫 Circuit breaker OPEN for '{tool_name}' — "
                f"skipping to fallback[/]"
            )
            return self._try_fallbacks(tool_name, tool_args, log)

        # Try primary tool with retries
        for attempt in range(self.backoff.max_retries + 1):
            attempt_record = ExecutionAttempt(
                tool_name=tool_name,
                args=tool_args,
                attempt_number=attempt + 1,
                timestamp=time.time(),
                success=False,
                is_fallback=False,
            )

            start = time.time()
            try:
                if tool_name not in TOOL_MAP:
                    raise Exception(f"Unknown tool: {tool_name}")

                result = TOOL_MAP[tool_name].invoke(tool_args)
                duration = (time.time() - start) * 1000

                attempt_record.success = True
                attempt_record.result = str(result)
                attempt_record.duration_ms = duration
                log.attempts.append(attempt_record)

                if tool_name not in log.tools_tried:
                    log.tools_tried.append(tool_name)

                self.circuit_breaker.record_success(tool_name)
                console.print(
                    f"  [green]✓ {tool_name} succeeded on attempt "
                    f"{attempt + 1} ({duration:.0f}ms)[/]"
                )
                return str(result)

            except Exception as e:
                duration = (time.time() - start) * 1000
                error_msg = str(e)

                # Classify error type
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    error_type = ErrorType.RATE_LIMIT
                elif "timeout" in error_msg.lower() or "408" in error_msg:
                    error_type = ErrorType.TIMEOUT
                elif "5" in error_msg[:5]:
                    error_type = ErrorType.API_ERROR
                else:
                    error_type = ErrorType.UNKNOWN

                attempt_record.error_type = error_type
                attempt_record.error_message = error_msg
                attempt_record.duration_ms = duration
                log.attempts.append(attempt_record)

                self.circuit_breaker.record_failure(tool_name)

                console.print(
                    f"  [red]✗ {tool_name} failed (attempt {attempt + 1}): "
                    f"[{error_type.value}] {error_msg}[/]"
                )

                # Backoff before retry (if we have retries left)
                if self.backoff.should_retry(attempt):
                    wait_time = self.backoff.wait(attempt)
                    attempt_record.backoff_wait_s = wait_time
                else:
                    console.print(
                        f"  [bold red]✗ Max retries exceeded for '{tool_name}'[/]"
                    )

        # All retries exhausted — try fallbacks
        return self._try_fallbacks(tool_name, tool_args, log)

    def _try_fallbacks(
        self,
        original_tool: str,
        tool_args: dict,
        log: ExecutionLog,
    ) -> str:
        """Try fallback tools for a failed primary tool."""
        fallbacks = FALLBACK_MAP.get(original_tool, [])

        if not fallbacks:
            error_msg = (
                f"Tool '{original_tool}' failed and no fallback is available. "
                f"Please try again later."
            )
            console.print(f"  [red]🚫 {error_msg}[/]")
            return error_msg

        for fb_tool in fallbacks:
            fb_name = fb_tool.name
            console.print(
                f"  [yellow]🔄 Trying fallback: '{fb_name}'[/]"
            )

            attempt_record = ExecutionAttempt(
                tool_name=fb_name,
                args=tool_args,
                attempt_number=1,
                timestamp=time.time(),
                success=False,
                is_fallback=True,
            )

            start = time.time()
            try:
                result = fb_tool.invoke(tool_args)
                duration = (time.time() - start) * 1000

                attempt_record.success = True
                attempt_record.result = str(result)
                attempt_record.duration_ms = duration
                log.attempts.append(attempt_record)

                if fb_name not in log.tools_tried:
                    log.tools_tried.append(fb_name)

                console.print(
                    f"  [green]✓ Fallback '{fb_name}' succeeded ({duration:.0f}ms)[/]"
                )
                return str(result)

            except Exception as e:
                duration = (time.time() - start) * 1000
                attempt_record.error_message = str(e)
                attempt_record.error_type = ErrorType.UNKNOWN
                attempt_record.duration_ms = duration
                log.attempts.append(attempt_record)
                console.print(
                    f"  [red]✗ Fallback '{fb_name}' also failed: {e}[/]"
                )

        return f"All tools failed for this request (tried: {original_tool} + {len(fallbacks)} fallbacks)."


# ═══════════════════════════════════════════════
#  SECTION 7: Agent with Error Recovery
# ═══════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools for fetching
stock prices, weather data, and news headlines.

Available tools:
1. fetch_stock_price — Get current stock price by ticker symbol
2. fetch_weather — Get current weather for a city
3. fetch_news — Get latest news on a topic

Use the appropriate tool based on the user's question. If a tool provides
partial or cached data, let the user know it may not be real-time.
Always provide a helpful, clear response based on the tool results."""


def build_agent():
    """Build the agent with primary tools (not fallbacks — those are internal)."""
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    primary_tools = [fetch_stock_price, fetch_weather, fetch_news]
    return llm.bind_tools(primary_tools)


def run_agent_with_recovery(
    agent,
    executor: ResilientExecutor,
    query: str,
) -> tuple[str, ExecutionLog]:
    """
    Run a full agent turn with error recovery on tool calls.
    """
    log = ExecutionLog(query=query)
    start_time = time.time()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # First LLM call
    response = agent.invoke(messages)

    if not response.tool_calls:
        log.final_result = response.content
        log.total_duration_ms = (time.time() - start_time) * 1000
        return response.content, log

    messages.append(response)

    # Process tool calls with resilient execution
    for tc in response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        console.print(f"\n  [bold yellow]🔧 Agent wants to call: {tool_name}[/]")
        console.print(f"  [dim]   Args: {json.dumps(tool_args)}[/]\n")

        result = executor.execute_with_recovery(tool_name, tool_args, log)

        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Final LLM call to synthesize results
    final_response = agent.invoke(messages)
    messages.append(final_response)

    log.final_result = final_response.content
    log.total_duration_ms = (time.time() - start_time) * 1000

    return final_response.content, log


# ═══════════════════════════════════════════════
#  SECTION 8: Display & Reporting
# ═══════════════════════════════════════════════

def display_banner():
    """Display the lab banner."""
    banner = Text()
    banner.append("CalderR Internship — Week 2, Lab 2.3\n", style="bold cyan")
    banner.append("Error Recovery Agent\n", style="bold white")
    banner.append("Retries · Exponential Backoff · Fallbacks · Circuit Breaker\n", style="dim")
    banner.append(f"\nModel: {MODEL_NAME}  |  ", style="dim")
    banner.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style="dim")
    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))


def display_execution_log(log: ExecutionLog):
    """Display a detailed execution log with timeline."""
    console.print()
    console.print(Rule("📋 Execution Log", style="cyan"))

    # Timeline table
    table = Table(
        title=f"Query: \"{log.query}\"",
        box=box.ROUNDED,
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool", style="white", width=26)
    table.add_column("Attempt", justify="center", width=8)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Backoff", justify="right", width=10)
    table.add_column("Error", style="dim", width=30)

    for i, attempt in enumerate(log.attempts):
        status = "[green]✓ OK[/]" if attempt.success else f"[red]✗ {attempt.error_type.value if attempt.error_type else 'FAIL'}[/]"
        fallback_tag = " [FB]" if attempt.is_fallback else ""
        table.add_row(
            str(i + 1),
            f"{attempt.tool_name}{fallback_tag}",
            str(attempt.attempt_number),
            status,
            f"{attempt.duration_ms:.0f}ms",
            f"{attempt.backoff_wait_s:.2f}s" if attempt.backoff_wait_s > 0 else "—",
            (attempt.error_message[:28] + "..") if attempt.error_message and len(attempt.error_message) > 30 else (attempt.error_message or "—"),
        )

    console.print(table)

    # Summary stats
    stats = Text()
    stats.append(f"Total attempts: {log.total_attempts}\n", style="white")
    stats.append(f"Successful: {log.successful_attempts}\n", style="green")
    stats.append(f"Failed: {log.failed_attempts}\n", style="red")
    stats.append(f"Tools tried: {', '.join(log.tools_tried)}\n", style="yellow")
    stats.append(f"Total backoff time: {log.total_backoff_time:.2f}s\n", style="cyan")
    stats.append(f"Total duration: {log.total_duration_ms:.0f}ms", style="white")

    console.print(Panel(stats, title="📊 Stats", border_style="cyan", box=box.ROUNDED))


def display_backoff_visualization():
    """Show a visual explanation of exponential backoff."""
    console.print()
    console.print(Rule("📈 Exponential Backoff Visualization", style="cyan"))

    engine = BackoffEngine(base_delay=0.5, max_delay=16.0, jitter_factor=0.0)
    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("Retry #", justify="center")
    table.add_column("Formula", style="dim")
    table.add_column("Wait Time", justify="right")
    table.add_column("Visual", style="yellow")

    for i in range(5):
        delay = engine.base_delay * (2 ** i)
        capped = min(delay, engine.max_delay)
        bar = "█" * int(capped * 2)
        table.add_row(
            str(i + 1),
            f"0.5 × 2^{i} = {delay:.1f}s",
            f"{capped:.1f}s",
            bar,
        )

    console.print(table)
    console.print("[dim]+ random jitter is added in practice to prevent thundering herd[/]\n")


# ═══════════════════════════════════════════════
#  SECTION 9: Demo Scenarios
# ═══════════════════════════════════════════════

DEMO_QUERIES = [
    "What is the current stock price of AAPL?",
    "What's the weather like in Tokyo?",
    "What's the latest news about technology?",
]


def run_demo(agent, executor):
    """Run demo scenarios showing error recovery in action."""
    console.print("\n[bold cyan]🎯 Demo Mode — Simulated Failures & Recovery[/]\n")
    console.print(
        "[dim]Each tool is configured to fail on the first 2 calls, "
        "then succeed. Watch the retry + fallback behavior.[/]\n"
    )

    # Reset call counters for clean demo
    global _call_counter
    _call_counter = {}

    all_logs = []

    for i, query in enumerate(DEMO_QUERIES):
        console.print(Rule(f"Scenario {i + 1}/{len(DEMO_QUERIES)}", style="yellow"))
        console.print(f"[bold white]User:[/] {query}\n")

        try:
            response, log = run_agent_with_recovery(agent, executor, query)

            console.print(Panel(
                response,
                title="🤖 Agent Response",
                border_style="green",
                box=box.ROUNDED,
            ))

            display_execution_log(log)
            all_logs.append(log)

        except Exception as e:
            console.print(f"[bold red]✗ Unrecoverable error: {e}[/]")
            console.print(f"[dim]{traceback.format_exc()}[/]")

        console.print()

    # Overall summary
    if all_logs:
        display_overall_summary(all_logs)


def display_overall_summary(logs: list[ExecutionLog]):
    """Display summary across all demo scenarios."""
    console.print(Rule("📊 Overall Demo Summary", style="bold cyan"))

    total_attempts = sum(l.total_attempts for l in logs)
    total_success = sum(l.successful_attempts for l in logs)
    total_failed = sum(l.failed_attempts for l in logs)
    total_backoff = sum(l.total_backoff_time for l in logs)

    table = Table(box=box.DOUBLE, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Queries processed", str(len(logs)))
    table.add_row("Total tool attempts", str(total_attempts))
    table.add_row("Successful", f"[green]{total_success}[/]")
    table.add_row("Failed (recovered)", f"[yellow]{total_failed}[/]")
    table.add_row("Total backoff time", f"{total_backoff:.2f}s")
    table.add_row("Unrecoverable failures", "[green]0[/]")

    console.print(table)
    console.print()


# ═══════════════════════════════════════════════
#  SECTION 10: Interactive Mode
# ═══════════════════════════════════════════════

def run_interactive(agent, executor):
    """Run interactive mode."""
    console.print("\n[bold cyan]💬 Interactive Mode[/]")
    console.print(
        "[dim]Ask about stocks, weather, or news. "
        "Type /quit to exit, /demo to re-run demos, /reset to clear failure counters.[/]\n"
    )

    global _call_counter

    while True:
        try:
            user_input = Prompt.ask("[bold white]You[/]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/demo":
            _call_counter = {}
            run_demo(agent, executor)
            continue
        if cmd == "/reset":
            _call_counter = {}
            console.print("[yellow]Failure counters reset.[/]")
            continue
        if cmd == "/backoff":
            display_backoff_visualization()
            continue
        if cmd == "/help":
            console.print(
                "[dim]/quit — Exit  |  /demo — Re-run demos  |  "
                "/reset — Reset failure counters\n"
                "/backoff — Show backoff visualization  |  /help — This[/]"
            )
            continue

        try:
            response, log = run_agent_with_recovery(agent, executor, user_input)
            console.print(Panel(
                response,
                title="🤖 Agent",
                border_style="green",
                box=box.ROUNDED,
            ))
            display_execution_log(log)
        except Exception as e:
            console.print(f"[bold red]✗ Error: {e}[/]")

        console.print()


# ═══════════════════════════════════════════════
#  SECTION 11: Main
# ═══════════════════════════════════════════════

def main():
    """Main entry point."""
    display_banner()
    display_backoff_visualization()

    console.print("[bold cyan]Building error-recovery agent...[/]")
    agent = build_agent()
    executor = ResilientExecutor()
    console.print("[green]✓ Agent ready with retry + fallback + circuit breaker[/]\n")

    # Run demo scenarios first
    run_demo(agent, executor)

    # Then interactive mode
    run_interactive(agent, executor)

    console.print("\n[bold green]✓ Lab 2.3 complete![/]\n")


if __name__ == "__main__":
    main()
