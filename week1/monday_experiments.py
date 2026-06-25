"""
CalderR Internship – Week 1, Monday
====================================
Applied Practice: Model & Temperature Experiments

What this file does:
  1. Runs the same prompt across 3 different Groq models and compares
     their speed, response length, and output style.
  2. Runs the same prompt at 4 different temperatures (0, 0.5, 1.0, 2.0)
     and documents how randomness changes the output.
  3. Prints everything with Rich so the terminal output is readable
     and worth keeping as a portfolio artifact.

Run:
    python week1/monday_experiments.py

    Follow the menu to choose which experiment to run.
"""

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.prompt import Prompt
from rich.markdown import Markdown

# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────

load_dotenv()
console = Console()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    console.print("[bold red]ERROR:[/] GROQ_API_KEY not found in .env file.")
    raise SystemExit(1)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

# The three models we'll compare. All available on Groq free tier.
MODELS = {
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B (Fast)",
        "color": "cyan",
        "description": "Groq's recommended fast model. Replaces Llama 3.1 8B.",
    },
    "qwen/qwen3.6-27b": {
        "label": "Qwen 3.6 – 27B (Balanced)",
        "color": "magenta",
        "description": "Alibaba's Qwen 3.6. Strong mid-size reasoning model.",
    },
    "openai/gpt-oss-120b": {
        "label": "GPT-OSS 120B (Powerful)",
        "color": "green",
        "description": "Groq's most powerful model. Replaces Llama 3.3 70B.",
    },
}

# The four temperatures we'll test.
TEMPERATURES = [0.0, 0.5, 1.0, 2.0]

# Temperature descriptions to help understand what each value means.
TEMP_DESCRIPTIONS = {
    0.0: "Deterministic – same output every run. Use for factual, precise tasks.",
    0.5: "Balanced – mostly consistent with slight variation. Good general default.",
    1.0: "Creative – noticeable variation between runs. Good for writing tasks.",
    2.0: "Chaotic – very high randomness. Often degrades coherence significantly.",
}


# ─────────────────────────────────────────────
#  Helper: build a LangChain chain
# ─────────────────────────────────────────────

def build_chain(model_name: str, temperature: float) -> object:
    """
    Constructs a simple LangChain LCEL chain:
        prompt | llm | output_parser

    Args:
        model_name:  Groq model identifier string.
        temperature: Sampling temperature (0.0 – 2.0).

    Returns:
        A runnable LangChain chain.
    """
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a concise AI engineering assistant. "
            "Keep every answer under 120 words unless asked otherwise.",
        ),
        ("user", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────
#  Helper: run one call and measure timing
# ─────────────────────────────────────────────

def run_call(chain, question: str) -> tuple[str, float]:
    """
    Invokes the chain with the given question and measures wall-clock time.

    Args:
        chain:    A built LangChain chain.
        question: The user question string.

    Returns:
        (response_text, elapsed_seconds)
    """
    start = time.perf_counter()
    response = chain.invoke({"question": question})
    elapsed = time.perf_counter() - start
    return response, elapsed


# ─────────────────────────────────────────────
#  Experiment 1: Model Comparison
# ─────────────────────────────────────────────

def experiment_model_comparison() -> None:
    """
    Sends the same prompt to all three Groq models and compares:
      - Response time (seconds)
      - Response length (word count)
      - Output quality (printed side by side)

    Key learning: same question, same temperature (0.7), different models
    → helps you understand trade-offs between speed and quality.
    """
    console.print()
    console.rule("[bold cyan]EXPERIMENT 1 — Model Comparison[/]")
    console.print(
        "\n[dim]Same prompt · Same temperature (0.7) · Three different models[/]\n"
    )

    question = (
        "Explain what an AI agent is and how it differs from a regular chatbot. "
        "Give one real-world use case."
    )

    console.print(
        Panel(
            f"[bold white]{question}[/]",
            title="[yellow]Prompt sent to all models[/]",
            border_style="yellow",
        )
    )
    console.print()

    # Collect results so we can show a summary table at the end.
    results: list[dict] = []

    for model_id, meta in MODELS.items():
        color = meta["color"]
        label = meta["label"]

        console.print(f"[{color}]▶ Calling {label} ...[/]")

        chain = build_chain(model_id, temperature=0.7)

        try:
            response, elapsed = run_call(chain, question)
        except Exception as exc:
            console.print(f"  [red]ERROR:[/] {exc}")
            continue

        word_count = len(response.split())

        # Print the response in a styled panel.
        console.print(
            Panel(
                response,
                title=f"[{color}]{label}[/]  [dim]{elapsed:.2f}s · {word_count} words[/]",
                border_style=color,
                padding=(1, 2),
            )
        )
        console.print()

        results.append({
            "model": label,
            "time": elapsed,
            "words": word_count,
            "color": color,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    console.rule("[bold white]Summary[/]")
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("Response Time", justify="right")
    table.add_column("Word Count", justify="right")
    table.add_column("Speed Rating", justify="center")

    for r in results:
        # Simple speed rating based on elapsed time.
        if r["time"] < 1.5:
            speed = "[green]⚡ Fast[/]"
        elif r["time"] < 3.5:
            speed = "[yellow]◎ Medium[/]"
        else:
            speed = "[red]🐢 Slow[/]"

        table.add_row(
            f"[{r['color']}]{r['model']}[/]",
            f"{r['time']:.2f}s",
            str(r["words"]),
            speed,
        )

    console.print(table)

    # ── What to learn from this ───────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold]Key Takeaway:[/]\n"
            "• [cyan]GPT-OSS 20B[/] is fastest — best for high-frequency, simple tasks.\n"
            "• [magenta]Qwen 3.6 27B[/] balances speed and quality in the mid-size range.\n"
            "• [green]GPT-OSS 120B[/] gives the richest, most powerful answers.\n\n"
            "[dim]In production, you pick the smallest model that meets your quality bar "
            "to save cost and latency.[/]",
            title="[bold yellow]What This Teaches You[/]",
            border_style="yellow",
        )
    )


# ─────────────────────────────────────────────
#  Experiment 2: Temperature Comparison
# ─────────────────────────────────────────────

def experiment_temperature_comparison() -> None:
    """
    Sends the same creative prompt to the same model at four temperatures.

    Key learning: temperature controls randomness in token sampling.
    0 = always picks the most probable token (deterministic).
    2 = samples very broadly (can become incoherent).
    """
    console.print()
    console.rule("[bold magenta]EXPERIMENT 2 — Temperature Comparison[/]")
    console.print(
        "\n[dim]Same prompt · Same model (openai/gpt-oss-120b) · Four temperatures[/]\n"
    )

    # A slightly creative question works best for showing temperature effects.
    # Factual questions look identical at all temperatures.
    question = (
        "Write a one-sentence tagline for a new AI-powered productivity tool "
        "aimed at software engineers."
    )

    console.print(
        Panel(
            f"[bold white]{question}[/]",
            title="[yellow]Prompt sent at all temperatures[/]",
            border_style="yellow",
        )
    )
    console.print()

    results: list[dict] = []

    for temp in TEMPERATURES:
        # Pick a color per temperature for visual clarity.
        color_map = {0.0: "cyan", 0.5: "green", 1.0: "yellow", 2.0: "red"}
        color = color_map[temp]

        console.print(f"[{color}]▶ Temperature = {temp}[/]  [dim]{TEMP_DESCRIPTIONS[temp]}[/]")

        chain = build_chain("openai/gpt-oss-120b", temperature=temp)

        try:
            response, elapsed = run_call(chain, question)
        except Exception as exc:
            console.print(f"  [red]ERROR:[/] {exc}")
            continue

        console.print(
            Panel(
                f"[white]{response}[/]",
                title=f"[{color}]temp={temp}[/]  [dim]{elapsed:.2f}s[/]",
                border_style=color,
                padding=(0, 2),
            )
        )
        console.print()

        results.append({
            "temp": temp,
            "response": response,
            "time": elapsed,
        })

    # ── Explanation table ──────────────────────────────────────────────────────
    console.rule("[bold white]Temperature Reference Guide[/]")
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    table.add_column("Temperature", justify="center", style="bold")
    table.add_column("Behaviour", style="white")
    table.add_column("Best For", style="dim")

    rows = [
        ("0.0", "[cyan]Deterministic[/] – same output every time",
         "Fact lookup, JSON extraction, code generation"),
        ("0.5", "[green]Balanced[/] – slight variation",
         "Q&A, summarisation, most general tasks"),
        ("1.0", "[yellow]Creative[/] – noticeable variation",
         "Writing, brainstorming, marketing copy"),
        ("2.0", "[red]Chaotic[/] – very high randomness",
         "Rarely useful; often degrades quality"),
    ]
    for temp_val, behaviour, use_for in rows:
        table.add_row(temp_val, behaviour, use_for)

    console.print(table)

    console.print()
    console.print(
        Panel(
            "[bold]Key Takeaway:[/]\n"
            "Temperature is a [cyan]sampling parameter[/] — it scales the probability "
            "distribution over the vocabulary before the model picks the next token.\n\n"
            "• At [cyan]temp=0[/], the highest-probability token is always chosen → "
            "deterministic.\n"
            "• At [yellow]temp=1[/], sampling follows the raw distribution → natural variation.\n"
            "• At [red]temp=2[/], the distribution is flattened → unlikely tokens get boosted "
            "→ incoherence.\n\n"
            "[dim]Rule of thumb: use 0 for structured outputs, 0.7 for most chat, "
            "1.0 for creative tasks.[/]",
            title="[bold yellow]What This Teaches You[/]",
            border_style="yellow",
        )
    )


# ─────────────────────────────────────────────
#  Experiment 3: Context Window Demo
# ─────────────────────────────────────────────

def experiment_context_window() -> None:
    """
    Demonstrates why the context window matters for agentic systems by
    simulating a multi-turn conversation where earlier turns get 'forgotten'
    once you exceed the window.

    This is intentionally simple — just a concept demo, not a real overflow.
    """
    console.print()
    console.rule("[bold green]EXPERIMENT 3 — Context Window Demo[/]")
    console.print(
        "\n[dim]Multi-turn conversation — watch how earlier context shapes later answers[/]\n"
    )

    # Build a raw LLM (no chain abstraction) so we can manage messages manually.
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        api_key=GROQ_API_KEY,
    )

    system = SystemMessage(content=(
        "You are a helpful assistant. Be concise — one or two sentences per reply."
    ))

    # Simulate a short 3-turn conversation manually.
    conversation_turns = [
        "My name is Ghafoor and I am an AI engineering intern.",
        "What technologies am I likely learning this week?",
        "Can you summarise what I told you about myself in one sentence?",
    ]

    messages = [system]

    for i, user_text in enumerate(conversation_turns, start=1):
        console.print(f"[bold cyan]Turn {i}[/]")
        console.print(
            Panel(user_text, title="[cyan]You[/]", border_style="cyan", padding=(0, 2))
        )

        messages.append(HumanMessage(content=user_text))

        try:
            ai_response = llm.invoke(messages)
            reply_text = ai_response.content
        except Exception as exc:
            console.print(f"[red]ERROR:[/] {exc}")
            break

        messages.append(AIMessage(content=reply_text))

        console.print(
            Panel(reply_text, title="[green]Assistant[/]", border_style="green", padding=(0, 2))
        )
        console.print(
            f"[dim]  Messages in context: {len(messages)} "
            f"(~{sum(len(m.content.split()) for m in messages)} words)[/]\n"
        )

    console.print(
        Panel(
            "[bold]Key Takeaway:[/]\n"
            "Every message — system, user, and assistant — is sent to the model "
            "on [bold]every turn[/].\n\n"
            "The model has no 'memory' between calls. The [cyan]context window[/] is "
            "the list of messages you pass in.\n\n"
            "• [green]GPT-OSS 20B[/] supports ~128 000 tokens context.\n"
            "• For a 10-hour conversation, you'd eventually hit this limit.\n"
            "• Agents handle this with [yellow]summarisation[/] or [yellow]vector "
            "memory[/] — topics for later weeks.",
            title="[bold yellow]What This Teaches You[/]",
            border_style="yellow",
        )
    )


# ─────────────────────────────────────────────
#  Main Menu
# ─────────────────────────────────────────────

def show_menu() -> None:
    console.print()
    console.print(
        Panel(
            "[bold white]CalderR Week 1 — Monday Experiments[/]\n"
            "[dim]AI Fundamentals & Agentic AI Foundations[/]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    console.print()
    console.print("  [bold cyan][1][/]  Model Comparison  "
                  "[dim](same prompt → 3 models → compare speed & quality)[/]")
    console.print("  [bold magenta][2][/]  Temperature Test  "
                  "[dim](same prompt → 4 temperatures → see randomness in action)[/]")
    console.print("  [bold green][3][/]  Context Window Demo  "
                  "[dim](multi-turn conversation → understand memory)[/]")
    console.print("  [bold yellow][4][/]  Run All Experiments  [dim](runs 1, 2, and 3)[/]")
    console.print("  [bold red][0][/]  Exit")
    console.print()


def main() -> None:
    while True:
        show_menu()
        choice = Prompt.ask(
            "[bold white]Choose an experiment[/]",
            choices=["0", "1", "2", "3", "4"],
            default="4",
        )

        if choice == "0":
            console.print("\n[dim]Goodbye.[/]\n")
            break
        elif choice == "1":
            experiment_model_comparison()
        elif choice == "2":
            experiment_temperature_comparison()
        elif choice == "3":
            experiment_context_window()
        elif choice == "4":
            experiment_model_comparison()
            experiment_temperature_comparison()
            experiment_context_window()

        console.print()
        console.rule("[dim]Experiment complete[/]")


if __name__ == "__main__":
    main()