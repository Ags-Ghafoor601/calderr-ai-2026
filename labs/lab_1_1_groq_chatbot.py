"""
CalderR Internship – Week 1, Lab 1.1
======================================
Groq CLI Chatbot

WHAT THIS LAB BUILDS:
---------------------
A fully interactive terminal chatbot powered by Groq that:
  • Maintains full conversation history across 10+ turns
  • Displays token usage (prompt + completion + total) per response
  • Supports /clear  → wipes history and starts fresh
  • Supports /help   → shows all commands
  • Supports /history → prints the full conversation so far
  • Supports /model  → switch model mid-conversation
  • Supports /exit   → quits cleanly
  • Uses Rich for a polished, readable terminal UI

WHAT THIS TEACHES YOU:
----------------------
  • How conversation memory works at the API level
    (you manually pass every prior message on every call)
  • What "token usage" means and why it matters for cost and limits
  • How system prompts shape the model's entire personality
  • How to build a CLI tool that feels professional

ARCHITECTURE:
  User Input (Rich Prompt)
      ↓
  Command Parser  (/clear, /exit, /help, /history, /model)
      ↓
  History Manager (appends user message to list)
      ↓
  Groq API Call   (sends full history every time)
      ↓
  Token Tracker   (reads usage from API response)
      ↓
  Rich Display    (renders assistant reply + token bar)

Run:
    python labs/lab_1_1_groq_chatbot.py
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich import box

# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────

load_dotenv()
console = Console()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    console.print("[bold red]ERROR:[/] GROQ_API_KEY not found in .env file.")
    sys.exit(1)

client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────
#  Available Models
#  Users can switch mid-conversation with /model
# ─────────────────────────────────────────────

AVAILABLE_MODELS = {
    "1": {
        "id":          "qwen/qwen3.6-27b",
        "name":        "Qwen 3.6 – 27B",
        "description": "Balanced quality and speed. Good for most tasks.",
        "color":       "magenta",
    },
    "2": {
        "id":          "openai/gpt-oss-120b",
        "name":        "GPT-OSS 120B",
        "description": "Most powerful. Best for complex reasoning.",
        "color":       "green",
    },
    "3": {
        "id":          "openai/gpt-oss-20b",
        "name":        "GPT-OSS 20B",
        "description": "Fastest responses. Great for quick Q&A.",
        "color":       "cyan",
    },
}

# Default model the chatbot starts with.
DEFAULT_MODEL_KEY = "1"


# ─────────────────────────────────────────────
#  System Prompt
#  This defines the chatbot's persona.
#  Changing this one block changes the entire
#  personality of the assistant — that's the
#  power of system prompts.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are CalderBot, an expert AI engineering assistant for the
CalderR Agentic AI Engineering Internship 2026.

Your personality:
- Precise and technically accurate — you never guess
- Encouraging to interns who are learning
- Concise by default — you answer in 2-4 sentences unless asked for detail
- You use bullet points only when listing 3+ items
- You always explain *why*, not just *what*

Your expertise covers:
- Python, LangChain, LangGraph, Groq API
- AI agents, RAG systems, prompt engineering
- FastAPI, Docker, ChromaDB, Pydantic
- Software engineering best practices

When you don't know something, say so clearly rather than guessing.
Keep responses focused and actionable."""


# ─────────────────────────────────────────────
#  Token Usage Tracker
#  Accumulates usage across the whole session
#  so the user can see total cost at the end.
# ─────────────────────────────────────────────

class TokenTracker:
    """
    Tracks token usage across the entire chat session.

    WHY THIS MATTERS:
    Every token costs money (on paid tiers) and consumes
    your context window. Tracking usage helps you:
      1. Understand how fast conversation history grows
      2. Know when you're approaching the model's context limit
      3. Estimate API costs in production
    """

    def __init__(self):
        self.total_prompt_tokens     = 0
        self.total_completion_tokens = 0
        self.total_tokens            = 0
        self.call_count              = 0
        # History of per-turn usage for the /history display
        self.turn_history: list[dict] = []

    def record(self, usage) -> dict:
        """Records usage from one API response and returns the turn stats."""
        pt = usage.prompt_tokens
        ct = usage.completion_tokens
        tt = usage.total_tokens

        self.total_prompt_tokens     += pt
        self.total_completion_tokens += ct
        self.total_tokens            += tt
        self.call_count              += 1

        turn = {
            "turn":       self.call_count,
            "prompt":     pt,
            "completion": ct,
            "total":      tt,
        }
        self.turn_history.append(turn)
        return turn

    def reset(self):
        """Called when /clear is used — resets all counters."""
        self.__init__()


# ─────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────

def print_welcome(model_info: dict) -> None:
    """Prints the startup banner."""
    console.print()
    console.print(
        Panel(
            "[bold white]CalderBot[/]  [dim]·  CalderR Agentic AI Internship 2026[/]\n\n"
            f"[dim]Model:[/]  [{model_info['color']}]{model_info['name']}[/]\n"
            "[dim]Type your message and press Enter. Commands start with /[/]\n\n"
            "[dim]/help for all commands · /exit to quit[/]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    console.print()


def print_help() -> None:
    """Displays the help table."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        title="[bold cyan]Available Commands[/]",
    )
    table.add_column("Command",     style="bold cyan", no_wrap=True)
    table.add_column("Description", style="white")

    commands = [
        ("/help",        "Show this help message"),
        ("/clear",       "Clear conversation history and start fresh"),
        ("/history",     "Show full conversation history with token counts"),
        ("/model",       "Switch to a different Groq model"),
        ("/stats",       "Show session token usage statistics"),
        ("/exit",        "Exit the chatbot"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()


def print_token_bar(turn: dict, model_color: str) -> None:
    """
    Prints a compact token usage line after each response.

    This teaches you to think about token consumption
    as a first-class concern — not an afterthought.
    """
    console.print(
        f"  [dim]tokens →  "
        f"prompt [yellow]{turn['prompt']:,}[/]  "
        f"completion [green]{turn['completion']:,}[/]  "
        f"total [{model_color}]{turn['total']:,}[/][/]"
    )
    console.print()


def print_user_message(text: str) -> None:
    """Renders the user's message."""
    console.print(
        Panel(
            f"[white]{text}[/]",
            title="[bold cyan]You[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def print_assistant_message(text: str, model_name: str, model_color: str,
                             elapsed: float) -> None:
    """
    Renders the assistant's reply as Markdown inside a Rich panel.
    Markdown rendering means **bold**, `code`, and bullet points
    all display correctly in the terminal.
    """
    console.print(
        Panel(
            Markdown(text),
            title=f"[{model_color}]{model_name}[/]  [dim]{elapsed:.2f}s[/]",
            border_style=model_color,
            padding=(1, 2),
        )
    )


def print_stats(tracker: TokenTracker) -> None:
    """Shows full session statistics."""
    if tracker.call_count == 0:
        console.print("[dim]No messages yet in this session.[/]\n")
        return

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title="[bold yellow]Session Statistics[/]",
    )
    table.add_column("Metric",  style="white")
    table.add_column("Value",   style="bold cyan", justify="right")

    table.add_row("Total turns",              str(tracker.call_count))
    table.add_row("Total prompt tokens",      f"{tracker.total_prompt_tokens:,}")
    table.add_row("Total completion tokens",  f"{tracker.total_completion_tokens:,}")
    table.add_row("Total tokens used",        f"{tracker.total_tokens:,}")
    table.add_row(
        "Avg tokens per turn",
        f"{tracker.total_tokens // tracker.call_count:,}"
    )

    console.print(table)
    console.print()


def print_history(messages: list[dict], tracker: TokenTracker) -> None:
    """
    Prints the full conversation history.

    This makes visible something most chatbots hide:
    every message in the list gets sent to the model on
    every single call. As the conversation grows, so does
    the token cost of each subsequent request.
    """
    if len(messages) <= 1:      # Only system prompt exists
        console.print("[dim]No conversation history yet.[/]\n")
        return

    console.print()
    console.rule("[bold yellow]Conversation History[/]")

    turn_idx = 0
    for msg in messages:
        if msg["role"] == "system":
            console.print(
                Panel(
                    f"[dim]{msg['content'][:120]}...[/]",
                    title="[dim]System Prompt[/]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )
        elif msg["role"] == "user":
            console.print(
                Panel(
                    f"[white]{msg['content']}[/]",
                    title="[cyan]You[/]",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )
        elif msg["role"] == "assistant":
            token_info = ""
            if turn_idx < len(tracker.turn_history):
                t = tracker.turn_history[turn_idx]
                token_info = f"  [dim]({t['total']:,} tokens)[/]"
                turn_idx += 1
            console.print(
                Panel(
                    Markdown(msg["content"]),
                    title=f"[green]Assistant[/]{token_info}",
                    border_style="green",
                    padding=(0, 2),
                )
            )

    console.print(
        f"\n[dim]Total messages in context: {len(messages)} "
        f"({tracker.total_tokens:,} tokens used so far)[/]\n"
    )


def show_model_picker() -> dict | None:
    """
    Displays the model selection menu and returns the chosen model dict,
    or None if the user cancels.
    """
    console.print()
    console.print("[bold yellow]Available Models:[/]\n")

    for key, model in AVAILABLE_MODELS.items():
        console.print(
            f"  [{model['color']}][{key}][/]  "
            f"[bold]{model['name']}[/]  "
            f"[dim]{model['description']}[/]"
        )

    console.print("  [dim][0]  Cancel[/]\n")

    choice = Prompt.ask(
        "[bold white]Choose model[/]",
        choices=["0", "1", "2", "3"],
        default="0",
    )

    if choice == "0":
        return None
    return AVAILABLE_MODELS[choice]


def show_thinking_spinner() -> Live:
    """
    Returns a Rich Live context that shows a spinner while
    waiting for the Groq API response.
    """
    spinner = Spinner("dots", text=" [dim]Thinking...[/]", style="green")
    return Live(spinner, console=console, refresh_per_second=12)


# ─────────────────────────────────────────────
#  Core Chat Function
# ─────────────────────────────────────────────

def send_message(
    messages:   list[dict],
    tracker:    TokenTracker,
    model_info: dict,
) -> str | None:
    """
    Sends the current message history to Groq and returns the reply.

    KEY INSIGHT: We send the ENTIRE messages list on every call.
    The model has no memory between calls — we create the illusion
    of memory by replaying the whole conversation each time.

    This is why long conversations get expensive: turn 10 sends
    10x more tokens than turn 1.

    Args:
        messages:   Full conversation history including system prompt.
        tracker:    Token tracker to record this call's usage.
        model_info: Dict with model id, name, color.

    Returns:
        The assistant's reply text, or None on error.
    """
    try:
        with show_thinking_spinner():
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model_info["id"],
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            elapsed = time.perf_counter() - start

        # Extract the reply text.
        reply = response.choices[0].message.content

        # Strip Qwen thinking tags if present.
        import re
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

        # Record token usage.
        turn_stats = tracker.record(response.usage)

        # Render the assistant's reply.
        print_assistant_message(
            reply,
            model_info["name"],
            model_info["color"],
            elapsed,
        )

        # Print the token bar beneath the reply.
        print_token_bar(turn_stats, model_info["color"])

        return reply

    except Exception as exc:
        console.print(
            Panel(
                f"[red]{exc}[/]",
                title="[red]API Error[/]",
                border_style="red",
            )
        )
        return None


# ─────────────────────────────────────────────
#  Command Handler
# ─────────────────────────────────────────────

def handle_command(
    command:    str,
    messages:   list[dict],
    tracker:    TokenTracker,
    model_info: dict,
) -> tuple[list[dict], TokenTracker, dict, bool]:
    """
    Processes a slash command and returns updated state.

    Returns:
        (messages, tracker, model_info, should_exit)
    """
    cmd = command.strip().lower()

    if cmd == "/exit":
        console.print()
        console.rule("[dim]Session Ended[/]")
        print_stats(tracker)
        console.print("[dim]Goodbye. Keep building.[/]\n")
        return messages, tracker, model_info, True

    elif cmd == "/help":
        print_help()

    elif cmd == "/clear":
        # Reset history but keep the system prompt.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        tracker.reset()
        console.print(
            Panel(
                "[white]Conversation cleared. History reset to zero.[/]\n"
                "[dim]The system prompt is preserved — the model's persona stays the same.[/]",
                title="[yellow]✓ Cleared[/]",
                border_style="yellow",
                padding=(0, 2),
            )
        )
        console.print()

    elif cmd == "/history":
        print_history(messages, tracker)

    elif cmd == "/stats":
        print_stats(tracker)

    elif cmd == "/model":
        chosen = show_model_picker()
        if chosen:
            model_info = chosen
            console.print(
                Panel(
                    f"[white]Switched to [{chosen['color']}]{chosen['name']}[/].\n"
                    "[dim]Conversation history is preserved — the new model sees all prior context.[/]",
                    title="[green]✓ Model Switched[/]",
                    border_style="green",
                    padding=(0, 2),
                )
            )
            console.print()
        else:
            console.print("[dim]Model unchanged.[/]\n")

    else:
        console.print(
            f"[yellow]Unknown command:[/] [white]{command}[/]\n"
            "[dim]Type /help to see all commands.[/]\n"
        )

    return messages, tracker, model_info, False


# ─────────────────────────────────────────────
#  Main Chat Loop
# ─────────────────────────────────────────────

def main() -> None:
    """
    The main conversation loop.

    Structure:
      1. Initialise state (messages, tracker, model)
      2. Print welcome
      3. Loop:
         a. Get user input
         b. If command → handle it
         c. If message → append to history, call API, append reply
         d. Continue until /exit
    """

    # ── Initialise state ──────────────────────────────────────────────────────
    model_info = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]
    tracker    = TokenTracker()

    # The messages list is our "memory".
    # It always starts with the system prompt.
    # Every user message and assistant reply gets appended here.
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # ── Welcome screen ────────────────────────────────────────────────────────
    print_welcome(model_info)

    # ── Conversation loop ─────────────────────────────────────────────────────
    while True:

        # ── Get input ─────────────────────────────────────────────────────────
        try:
            # Show turn number and token count in the prompt.
            turn_num   = tracker.call_count + 1
            token_info = (
                f"[dim]{tracker.total_tokens:,} tokens[/]  "
                if tracker.call_count > 0
                else ""
            )
            user_input = Prompt.ask(
                f"\n[bold cyan]You[/] [dim](turn {turn_num})[/]  {token_info}"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D exits cleanly.
            console.print()
            console.rule("[dim]Session Ended[/]")
            print_stats(tracker)
            console.print("[dim]Goodbye. Keep building.[/]\n")
            break

        # ── Empty input ───────────────────────────────────────────────────────
        if not user_input:
            console.print("[dim]Type a message or /help for commands.[/]")
            continue

        # ── Command routing ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            messages, tracker, model_info, should_exit = handle_command(
                user_input, messages, tracker, model_info
            )
            if should_exit:
                break
            continue

        # ── Normal message ─────────────────────────────────────────────────────
        # 1. Render what the user typed.
        print_user_message(user_input)

        # 2. Append to history BEFORE calling the API.
        #    This is how the model knows what the user just said.
        messages.append({"role": "user", "content": user_input})

        # 3. Call the API with the full history.
        reply = send_message(messages, tracker, model_info)

        # 4. Append the assistant's reply to history.
        #    On the next turn, the model will see this reply as context.
        if reply:
            messages.append({"role": "assistant", "content": reply})

        # 5. Warn if context is getting long (over 80% of typical limit).
        #    This mimics what production agents do with memory management.
        if tracker.total_tokens > 80_000:
            console.print(
                Panel(
                    "[yellow]Context window is getting large "
                    f"({tracker.total_tokens:,} tokens).[/]\n"
                    "[dim]Consider /clear to reset history "
                    "or the model may start losing early context.[/]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )


if __name__ == "__main__":
    main()