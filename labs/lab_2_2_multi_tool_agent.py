"""
CalderR Internship – Week 2, Lab 2.2
======================================
Multi-Tool Research Agent — 5+ Tools with Intelligent Routing

WHAT THIS LAB BUILDS:
---------------------
An interactive agent powered by LangChain + Groq that has access to
5+ callable tools and intelligently routes user queries to the
correct tool:

  1. web_search_mock   → Searches a mock knowledge base of web results
  2. calculate         → Evaluates mathematical expressions safely
  3. get_current_date  → Returns the current date/time in any timezone
  4. summarize         → Summarizes long text into bullet points
  5. classify_sentiment → Classifies text as positive/negative/neutral
  6. convert_units     → Converts between units (bonus 6th tool)

WHAT THIS TEACHES YOU:
----------------------
  • How to define tools using LangChain's @tool decorator
  • How tool schemas (OpenAI format) are generated from Python functions
  • How ChatGroq.bind_tools() attaches tool schemas to the model
  • How the agent's tool-calling loop works (invoke → tool_call → result)
  • How to build a complete ReAct-style tool-calling agent

ARCHITECTURE:
  User Query
      ↓
  ChatGroq (with bound tools)
      ↓
  Tool Selection (model decides which tool to call)
      ↓
  Tool Execution (Python function runs)
      ↓
  Result fed back to model
      ↓
  Final Answer (natural language)

Run:
    python labs/lab_2_2_multi_tool_agent.py
"""

import os
import sys
import math
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

# Fix Windows console encoding (cp1252 cannot handle Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
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
#  SECTION 1: Tool Definitions (6 Tools)
# ═══════════════════════════════════════════════

# ── Tool 1: Mock Web Search ──────────────────

MOCK_WEB_DATA = {
    "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple paradigms. Python 3.12 is the latest stable release as of 2024. It is widely used in AI, web development, data science, and automation.",
    "langchain": "LangChain is a framework for developing applications powered by large language models (LLMs). It provides tools for prompt management, chains, agents, and memory. Founded by Harrison Chase in 2022, it has become the most popular LLM application framework.",
    "groq": "Groq is an AI inference company that builds custom hardware (Language Processing Units / LPUs) for running LLMs at extremely high speed. Their API provides the fastest inference speeds in the industry, supporting models like Llama 3 and Mixtral.",
    "pydantic": "Pydantic is a data validation library for Python. Version 2 (2023) was a complete rewrite in Rust for 5-50x performance improvement. It uses Python type hints to validate data, serialize/deserialize JSON, and generate JSON schemas.",
    "fastapi": "FastAPI is a modern, high-performance web framework for building APIs with Python based on type hints. Created by Sebastián Ramírez, it automatically generates OpenAPI documentation and uses Pydantic for request/response validation.",
    "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. Key subfields include Machine Learning, Deep Learning, Natural Language Processing, and Computer Vision. The field was founded at the Dartmouth Conference in 1956.",
    "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, primarily burning fossil fuels, have been the main driver since the 1800s. The Paris Agreement (2015) aims to limit warming to 1.5°C above pre-industrial levels.",
    "machine learning": "Machine Learning is a subset of AI where systems learn from data without explicit programming. Key approaches include supervised learning, unsupervised learning, and reinforcement learning. Neural networks, especially deep learning, have driven recent breakthroughs.",
}


@tool
def web_search_mock(query: str) -> str:
    """Search the web for information about a topic. Returns relevant search results.
    Use this when the user asks about a topic, concept, technology, or current events."""
    query_lower = query.lower().strip()
    results = []
    for key, value in MOCK_WEB_DATA.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            results.append(f"[{key.title()}]: {value}")
    if results:
        return "\n\n".join(results)
    return f"No results found for '{query}'. Try a different search term."


# ── Tool 2: Calculator ───────────────────────

SAFE_MATH_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sum": sum, "len": len,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "log2": math.log2, "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial,
}


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports arithmetic (+, -, *, /, **, %), functions (sqrt, sin, cos, log),
    and constants (pi, e). Use this when the user asks to compute something."""
    try:
        # Sanitize: only allow safe characters and function names
        sanitized = expression.strip()
        # Evaluate safely
        result = eval(sanitized, {"__builtins__": {}}, SAFE_MATH_NAMES)
        if isinstance(result, float):
            # Format nicely
            if result == int(result) and abs(result) < 1e15:
                return f"{int(result)}"
            return f"{result:.6g}"
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


# ── Tool 3: Current Date/Time ────────────────

TIMEZONE_OFFSETS = {
    "utc": 0, "gmt": 0, "est": -5, "edt": -4, "cst": -6,
    "cdt": -5, "mst": -7, "mdt": -6, "pst": -8, "pdt": -7,
    "ist": 5.5, "jst": 9, "cet": 1, "eet": 2, "aest": 10,
    "pkt": 5, "bst": 1, "sgt": 8, "hkt": 8, "kst": 9,
}


@tool
def get_current_date(timezone_name: str = "utc") -> str:
    """Get the current date and time in a specific timezone.
    Supported timezones: UTC, GMT, EST, EDT, CST, CDT, MST, MDT, PST, PDT,
    IST, JST, CET, EET, AEST, PKT, BST, SGT, HKT, KST.
    Use this when the user asks what time or date it is."""
    tz_key = timezone_name.lower().strip()
    offset_hours = TIMEZONE_OFFSETS.get(tz_key, None)

    if offset_hours is None:
        return f"Unknown timezone '{timezone_name}'. Supported: {', '.join(TIMEZONE_OFFSETS.keys())}"

    # Handle fractional hours (e.g., IST = +5:30)
    hours = int(offset_hours)
    minutes = int((offset_hours - hours) * 60)
    tz = timezone(timedelta(hours=hours, minutes=minutes))
    now = datetime.now(tz)

    return (
        f"Current date and time in {timezone_name.upper()}:\n"
        f"  Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"  Time: {now.strftime('%I:%M:%S %p')}\n"
        f"  ISO:  {now.isoformat()}"
    )


# ── Tool 4: Text Summarizer ─────────────────

@tool
def summarize(text: str) -> str:
    """Summarize a given text into concise bullet points.
    Use this when the user provides a long passage and asks for a summary.
    The tool extracts key sentences and presents them as bullet points."""
    if not text or not text.strip():
        return "Error: No text provided to summarize."

    # Simple extractive summarization (sentence scoring)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "Text is too short to summarize meaningfully."

    # Score sentences by position and length
    scored = []
    for i, sent in enumerate(sentences):
        word_count = len(sent.split())
        # Favor early sentences and medium-length ones
        position_score = 1.0 / (i + 1)
        length_score = min(word_count / 20.0, 1.0)
        score = position_score * 0.6 + length_score * 0.4
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top sentences (up to 5)
    top_n = min(5, len(scored))
    key_points = [s[1] for s in scored[:top_n]]

    total_words = len(text.split())
    summary_words = sum(len(p.split()) for p in key_points)

    result = "📝 Summary:\n"
    for i, point in enumerate(key_points, 1):
        result += f"  {i}. {point}.\n"
    result += f"\n[Compressed {total_words} words → {summary_words} words ({summary_words/total_words*100:.0f}% of original)]"

    return result


# ── Tool 5: Sentiment Classifier ────────────

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "good",
    "love", "happy", "joy", "excited", "brilliant", "perfect", "best",
    "awesome", "outstanding", "beautiful", "incredible", "superb", "delighted",
    "pleased", "grateful", "impressive", "success", "win", "positive",
    "recommend", "enjoy", "helpful", "friendly", "efficient",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "angry",
    "disappointed", "poor", "worst", "fail", "ugly", "broken", "useless",
    "annoying", "frustrating", "mistake", "problem", "issue", "wrong",
    "slow", "expensive", "waste", "boring", "confusing", "difficult",
    "complicated", "error", "crash", "bug",
}


@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of a text as positive, negative, or neutral.
    Returns the sentiment label along with a confidence score.
    Use this when the user asks about the sentiment or tone of a piece of text."""
    if not text or not text.strip():
        return "Error: No text provided for sentiment analysis."

    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        sentiment = "NEUTRAL"
        confidence = 0.50
        explanation = "No strong sentiment indicators found."
    elif pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = min(0.95, 0.5 + (pos_count - neg_count) / (total + 2) * 0.5)
        explanation = f"Found {pos_count} positive and {neg_count} negative indicators."
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = min(0.95, 0.5 + (neg_count - pos_count) / (total + 2) * 0.5)
        explanation = f"Found {neg_count} negative and {pos_count} positive indicators."
    else:
        sentiment = "MIXED"
        confidence = 0.45
        explanation = f"Equal positive ({pos_count}) and negative ({neg_count}) indicators."

    return (
        f"Sentiment Analysis:\n"
        f"  Label:      {sentiment}\n"
        f"  Confidence: {confidence:.0%}\n"
        f"  Reason:     {explanation}\n"
        f"  Positive words found: {', '.join(words & POSITIVE_WORDS) or 'none'}\n"
        f"  Negative words found: {', '.join(words & NEGATIVE_WORDS) or 'none'}"
    )


# ── Tool 6: Unit Converter (Bonus) ──────────

UNIT_CONVERSIONS = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
    ("meters", "feet"): 3.28084,
    ("feet", "meters"): 0.3048,
    ("liters", "gallons"): 0.264172,
    ("gallons", "liters"): 3.78541,
    ("cm", "inches"): 0.393701,
    ("inches", "cm"): 2.54,
}


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value from one unit to another.
    Supports: km↔miles, kg↔lbs, celsius↔fahrenheit, meters↔feet,
    liters↔gallons, cm↔inches.
    Use this when the user asks to convert measurements between units."""
    key = (from_unit.lower().strip(), to_unit.lower().strip())
    conversion = UNIT_CONVERSIONS.get(key)

    if conversion is None:
        supported = ", ".join(f"{a}→{b}" for a, b in UNIT_CONVERSIONS.keys())
        return f"Unsupported conversion: {from_unit} → {to_unit}. Supported: {supported}"

    if callable(conversion):
        result = conversion(value)
    else:
        result = value * conversion

    return f"{value} {from_unit} = {result:.4g} {to_unit}"


# ═══════════════════════════════════════════════
#  SECTION 2: Agent Setup
# ═══════════════════════════════════════════════

ALL_TOOLS = [
    web_search_mock,
    calculate,
    get_current_date,
    summarize,
    classify_sentiment,
    convert_units,
]

SYSTEM_PROMPT = """You are a helpful research assistant with access to 6 tools:

1. **web_search_mock** — Search for information about topics, technologies, concepts
2. **calculate** — Evaluate math expressions (arithmetic, trig, log, etc.)
3. **get_current_date** — Get current date/time in any timezone
4. **summarize** — Summarize long text into bullet points
5. **classify_sentiment** — Analyze sentiment of text (positive/negative/neutral)
6. **convert_units** — Convert measurements (km↔miles, kg↔lbs, °C↔°F, etc.)

ROUTING RULES:
- If the user asks about a topic or wants information → use web_search_mock
- If the user asks to compute, calculate, or do math → use calculate
- If the user asks about the current date or time → use get_current_date
- If the user provides text and asks for a summary → use summarize
- If the user asks about sentiment or tone → use classify_sentiment
- If the user asks to convert units/measurements → use convert_units

Always use tools when appropriate. Provide clear, helpful answers based on tool results.
If no tool is needed, respond directly."""


def build_agent():
    """Build the multi-tool agent with bound tools."""
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return llm_with_tools


def execute_tool_call(tool_call: dict) -> str:
    """Execute a single tool call and return the result."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Find the matching tool
    tool_map = {t.name: t for t in ALL_TOOLS}
    if tool_name not in tool_map:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        result = tool_map[tool_name].invoke(tool_args)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent_turn(agent, messages: list) -> tuple[str, list[dict]]:
    """
    Run a single agent turn: invoke the model, process any tool calls,
    and return the final response along with a log of tool calls made.
    """
    tool_log = []

    # First LLM call
    response = agent.invoke(messages)

    # If no tool calls, return the text response directly
    if not response.tool_calls:
        return response.content, tool_log

    # Process tool calls (may be multiple in parallel)
    messages.append(response)

    for tc in response.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        console.print(f"  [bold yellow]🔧 Calling tool:[/] {tool_name}")
        console.print(f"  [dim]   Args: {json.dumps(tool_args, indent=2)}[/]")

        result = execute_tool_call(tc)
        tool_log.append({
            "tool": tool_name,
            "args": tool_args,
            "result": result[:200] + "..." if len(result) > 200 else result,
        })

        console.print(f"  [green]   ✓ Result received ({len(result)} chars)[/]")

        # Add tool result as ToolMessage
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Second LLM call to synthesize tool results into final answer
    final_response = agent.invoke(messages)
    messages.append(final_response)

    return final_response.content, tool_log


# ═══════════════════════════════════════════════
#  SECTION 3: Display & UI
# ═══════════════════════════════════════════════

def display_banner():
    """Display the lab banner."""
    banner = Text()
    banner.append("CalderR Internship — Week 2, Lab 2.2\n", style="bold cyan")
    banner.append("Multi-Tool Research Agent\n", style="bold white")
    banner.append("6 Tools · Intelligent Routing · Interactive CLI\n", style="dim")
    banner.append(f"\nModel: {MODEL_NAME}  |  ", style="dim")
    banner.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style="dim")
    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))


def display_tool_catalog():
    """Display the available tools in a table."""
    table = Table(
        title="🧰 Available Tools",
        box=box.ROUNDED,
        title_style="bold yellow",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool", style="bold white", width=22)
    table.add_column("Description", style="dim", width=50)

    tool_info = [
        ("1", "web_search_mock", "Search for topic info"),
        ("2", "calculate", "Math expressions"),
        ("3", "get_current_date", "Date/time in any timezone"),
        ("4", "summarize", "Summarize long text"),
        ("5", "classify_sentiment", "Sentiment analysis"),
        ("6", "convert_units", "Unit conversion"),
    ]
    for num, name, desc in tool_info:
        table.add_row(num, name, desc)

    console.print(table)


def display_tool_schemas():
    """Display the OpenAI-format tool schemas."""
    console.print(Rule("📋 Tool Schemas (OpenAI Format)", style="dim"))
    for t in ALL_TOOLS:
        schema = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema() if t.args_schema else {},
            },
        }
        console.print(Panel(
            json.dumps(schema, indent=2),
            title=f"Schema: {t.name}",
            border_style="dim",
            box=box.SIMPLE,
        ))


# ═══════════════════════════════════════════════
#  SECTION 4: Demo Queries
# ═══════════════════════════════════════════════

DEMO_QUERIES = [
    "What is LangChain and what is it used for?",
    "What's the square root of 144 plus 3 to the power of 4?",
    "What time is it right now in PST?",
    "Please summarize: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems use algorithms to identify patterns in data and make decisions with minimal human intervention. Key approaches include supervised learning where models learn from labeled data, unsupervised learning which finds hidden patterns, and reinforcement learning where agents learn through trial and error.",
    "What's the sentiment of this review: This product is absolutely amazing! Best purchase I've ever made. The quality is outstanding and the customer service was incredibly helpful and friendly.",
    "Convert 100 kilometers to miles",
]


def run_demo_mode(agent):
    """Run through demo queries to showcase all tools."""
    console.print("\n[bold cyan]🎯 Running Demo Mode — Testing All 6 Tools[/]\n")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for i, query in enumerate(DEMO_QUERIES):
        console.print(Rule(f"Demo Query {i + 1}/{len(DEMO_QUERIES)}", style="yellow"))
        console.print(f"[bold white]User:[/] {query}\n")

        # Use fresh message list per query for clean routing
        turn_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

        try:
            response, tool_log = run_agent_turn(agent, turn_messages)

            if tool_log:
                for entry in tool_log:
                    console.print(
                        f"  [dim]Tool used: {entry['tool']} → "
                        f"{entry['result'][:100]}...[/]"
                    )

            console.print(Panel(
                response,
                title="🤖 Agent Response",
                border_style="green",
                box=box.ROUNDED,
            ))
        except Exception as e:
            console.print(f"[bold red]✗ Error: {e}[/]")

        console.print()


# ═══════════════════════════════════════════════
#  SECTION 5: Interactive Mode
# ═══════════════════════════════════════════════

def run_interactive_mode(agent):
    """Run the agent in interactive mode."""
    console.print("\n[bold cyan]💬 Interactive Mode[/]")
    console.print("[dim]Type your questions. Type /quit to exit, /demo for demo mode.[/]\n")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

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
            run_demo_mode(agent)
            continue
        if cmd == "/tools":
            display_tool_catalog()
            continue
        if cmd == "/schemas":
            display_tool_schemas()
            continue
        if cmd == "/clear":
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            console.print("[yellow]History cleared.[/]")
            continue
        if cmd == "/help":
            console.print(
                "[dim]/quit — Exit  |  /demo — Run demos  |  /tools — List tools\n"
                "/schemas — Show tool schemas  |  /clear — Clear history  |  /help — This[/]"
            )
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            response, tool_log = run_agent_turn(agent, messages)

            console.print(Panel(
                response,
                title="🤖 Agent",
                border_style="green",
                box=box.ROUNDED,
            ))
        except Exception as e:
            console.print(f"[bold red]✗ Error: {e}[/]")
            # Remove the failed user message to keep history clean
            messages.pop()

        console.print()


# ═══════════════════════════════════════════════
#  SECTION 6: Main
# ═══════════════════════════════════════════════

def main():
    """Main entry point."""
    display_banner()
    display_tool_catalog()

    console.print("\n[bold cyan]Building multi-tool agent...[/]")
    agent = build_agent()
    console.print("[green]✓ Agent ready with 6 tools bound[/]\n")

    # Run demo first, then interactive
    run_demo_mode(agent)
    run_interactive_mode(agent)

    console.print("\n[bold green]✓ Lab 2.2 complete![/]\n")


if __name__ == "__main__":
    main()
