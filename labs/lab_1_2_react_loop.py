"""
CalderR Internship – Week 1, Lab 1.2
======================================
Manual ReAct Loop — Pure Python, No Frameworks

WHAT IS ReAct?
--------------
ReAct = Reasoning + Acting. It's a pattern where an LLM doesn't just
answer directly — instead it:

  1. THINKS  → "What do I need to do here?"
  2. ACTS    → Calls a tool to get real information
  3. OBSERVES → Reads the tool's output
  4. LOOPS   → Thinks again based on what it learned
  5. ANSWERS → Only when it has enough information

This loop is the foundation of every AI agent ever built. LangChain,
LangGraph, AutoGen — they all implement this same pattern under the hood.
By building it manually here, you understand what those frameworks are
actually doing.

THE THREE TOOLS:
  - search_facts(query)       → searches a hardcoded knowledge base
  - calculate(expression)     → evaluates a math expression safely
  - text_analyze(text)        → counts words, chars, sentences

ARCHITECTURE (no framework — pure Python + Groq SDK):

  User Input
      ↓
  [ THINK ]  LLM reasons about what to do next
      ↓
  [ ACT ]    Parser extracts tool name + input from LLM output
      ↓
  [ OBSERVE ] Tool runs, result is collected
      ↓
  [ LOOP ]   Result fed back to LLM as context → repeat
      ↓
  [ ANSWER ] LLM outputs "Final Answer:" → loop ends

Run:
    python labs/lab_1_2_react_loop.py
"""

import os
import re
import math
import json
from dotenv import load_dotenv
from groq import Groq                   # Direct Groq SDK — no LangChain
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.prompt import Prompt
from rich import box

# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────

load_dotenv()
console = Console()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The model we use for the agent's reasoning.
# Using the fast model so the ReAct loop feels snappy.
AGENT_MODEL = "qwen/qwen3.6-27b"

# Maximum number of Thought→Act→Observe cycles before we force-stop.
# This prevents infinite loops — a critical safety pattern in real agents.
MAX_ITERATIONS = 6


# ─────────────────────────────────────────────
#  TOOL DEFINITIONS
#  Each tool is just a plain Python function.
#  The agent (LLM) decides which one to call.
# ─────────────────────────────────────────────

# A hardcoded "knowledge base" the search tool looks through.
# In Week 3 (RAG), this gets replaced with a real vector database.
KNOWLEDGE_BASE = {
    "python": (
        "Python is a high-level, interpreted programming language created by "
        "Guido van Rossum and released in 1991. It emphasises code readability "
        "and simplicity. Python supports multiple paradigms: procedural, "
        "object-oriented, and functional programming."
    ),
    "langchain": (
        "LangChain is an open-source framework for building applications powered "
        "by large language models. It provides abstractions for chains, agents, "
        "memory, and retrieval. The LangChain Expression Language (LCEL) uses "
        "the pipe operator | to compose steps."
    ),
    "groq": (
        "Groq is an AI inference company that builds custom chips called LPUs "
        "(Language Processing Units). GroqCloud offers a free API tier for "
        "running open-source models like Llama and Qwen at very high speed — "
        "often 300-1000 tokens per second."
    ),
    "react": (
        "ReAct (Reasoning + Acting) is a prompting pattern introduced in a 2022 "
        "paper by Yao et al. The agent interleaves Thought steps (reasoning) with "
        "Action steps (tool calls) and Observation steps (tool results). This "
        "enables dynamic decision-making beyond a single LLM call."
    ),
    "agent": (
        "An AI agent is a system that perceives its environment, makes decisions, "
        "and takes actions to achieve a goal. Unlike a simple chatbot that only "
        "responds to input, an agent can plan, use tools, maintain memory, and "
        "loop until it completes a task."
    ),
    "llm": (
        "A Large Language Model (LLM) is a neural network trained on massive "
        "text datasets to predict the next token. LLMs like GPT, Llama, and "
        "Qwen use the Transformer architecture (attention mechanism). They are "
        "the reasoning engine inside every modern AI agent."
    ),
    "transformer": (
        "The Transformer architecture was introduced in the 2017 paper 'Attention "
        "Is All You Need' by Vaswani et al. at Google. It replaced recurrent "
        "neural networks with self-attention mechanisms, enabling parallel "
        "training on huge datasets and scaling to billions of parameters."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with "
        "an LLM. Instead of relying purely on the model's training data, RAG "
        "fetches relevant documents from a knowledge base and injects them into "
        "the prompt. This reduces hallucinations and keeps knowledge up to date."
    ),
    "calder": (
        "CalderR is an AI engineering company with the tagline 'Rethinking how "
        "work works'. The 2026 Agentic AI Engineering Internship is a 10-week "
        "programme (200 hours total) focused on building production-ready "
        "agentic AI systems using Python, LangChain, and Groq."
    ),
    "vector database": (
        "A vector database stores embeddings — numerical representations of text "
        "or images. It supports similarity search: given a query embedding, it "
        "finds the most similar stored vectors. ChromaDB, Pinecone, and Weaviate "
        "are popular vector databases used in RAG systems."
    ),
}


def search_facts(query: str) -> str:
    """
    TOOL 1: Fact Search
    -------------------
    Searches the knowledge base for entries matching the query.
    Returns the matching fact, or a not-found message.

    In production this would call a vector DB or a web search API.
    For now it's a simple keyword match so the logic stays visible.
    """
    query_lower = query.lower().strip()

    # Try exact key match first.
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower or query_lower in key:
            return f"[From knowledge base — '{key}']\n{value}"

    # Fuzzy: check if any word in the query appears in any key.
    query_words = set(query_lower.split())
    for key, value in KNOWLEDGE_BASE.items():
        key_words = set(key.split())
        if query_words & key_words:          # set intersection
            return f"[From knowledge base — '{key}']\n{value}"

    return (
        f"No fact found for '{query}'. "
        f"Available topics: {', '.join(KNOWLEDGE_BASE.keys())}."
    )


def calculate(expression: str) -> str:
    """
    TOOL 2: Calculator
    ------------------
    Safely evaluates a mathematical expression using Python's eval().

    We restrict eval() to math functions only — this is important.
    Allowing arbitrary eval() is a security vulnerability in real systems.
    The 'allowed_names' dict is our sandbox.
    """
    # Whitelist: only these names are available inside eval().
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        # All math module functions
        **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
    }

    try:
        # Clean the expression — remove any text wrapping.
        clean = expression.strip().strip("'\"")
        result = eval(clean, {"__builtins__": {}}, allowed_names)
        return f"Result of '{clean}' = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def text_analyze(text: str) -> str:
    """
    TOOL 3: Text Analyser
    ---------------------
    Returns basic statistics about a piece of text.
    Useful for agents that process documents or user input.
    """
    text = text.strip().strip("'\"")

    if not text:
        return "Error: No text provided."

    words      = text.split()
    sentences  = re.split(r'[.!?]+', text)
    sentences  = [s.strip() for s in sentences if s.strip()]
    chars      = len(text)
    chars_nsp  = len(text.replace(" ", ""))
    avg_word   = round(sum(len(w) for w in words) / len(words), 1) if words else 0

    return (
        f"Text analysis results:\n"
        f"  • Word count:        {len(words)}\n"
        f"  • Character count:   {chars} (with spaces), {chars_nsp} (without)\n"
        f"  • Sentence count:    {len(sentences)}\n"
        f"  • Avg word length:   {avg_word} characters\n"
        f"  • Unique words:      {len(set(w.lower().strip('.,!?') for w in words))}"
    )


# ─────────────────────────────────────────────
#  TOOL REGISTRY
#  This maps tool names (strings the LLM writes)
#  to the actual Python functions above.
# ─────────────────────────────────────────────

TOOLS = {
    "search_facts": search_facts,
    "calculate":    calculate,
    "text_analyze": text_analyze,
}

# Human-readable description sent to the LLM in the system prompt.
# The quality of this description directly determines how well the
# agent picks the right tool — this is prompt engineering in practice.
TOOL_DESCRIPTIONS = """
You have access to exactly three tools. Use them whenever you need information
or need to compute something. Never guess at facts or math — always use a tool.

TOOL 1 — search_facts(query)
  Use when: the user asks about a concept, technology, or topic.
  Input:    a short keyword or phrase (e.g. "python", "LLM", "RAG")
  Returns:  a paragraph of factual information from the knowledge base.

TOOL 2 — calculate(expression)
  Use when: the user asks for any mathematical calculation.
  Input:    a valid Python math expression (e.g. "2 ** 10", "sqrt(144)", "15 * 8 + 3")
  Returns:  the numeric result.

TOOL 3 — text_analyze(text)
  Use when: the user asks you to analyse, count words in, or examine a piece of text.
  Input:    the text to analyse (wrap in single quotes if it contains spaces)
  Returns:  word count, character count, sentence count, and average word length.
"""


# ─────────────────────────────────────────────
#  SYSTEM PROMPT
#  This is the most important part of any agent.
#  It defines the agent's format, personality,
#  and decision-making process.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a ReAct agent — an AI that Reasons and Acts to answer questions.

You MUST follow this exact format for every response. No exceptions:

Thought: <your reasoning about what to do next>
Action: <tool name — one of: search_facts, calculate, text_analyze>
Action Input: <the input to pass to the tool>

After you receive an Observation, continue with another Thought/Action/Action Input
OR write your final answer in this exact format:

Thought: I now have enough information to answer.
Final Answer: <your complete answer to the user's question>

RULES:
- Always start with a Thought.
- Never skip the Action step if you need information.
- Never make up facts — use search_facts instead.
- Never calculate in your head — use calculate instead.
- The Final Answer must directly and completely answer the user's original question.
- Keep Thoughts concise (one or two sentences).

{TOOL_DESCRIPTIONS}
"""


# ─────────────────────────────────────────────
#  RESPONSE PARSER
#  The LLM outputs plain text. We parse it to
#  extract: Action name + Action Input
#  OR detect when it has written a Final Answer.
# ─────────────────────────────────────────────

def parse_llm_response(response_text: str) -> dict:
    """
    Parses the LLM's raw text output into a structured dict.

    Returns one of three shapes:
      {"type": "action",       "tool": "...", "input": "..."}
      {"type": "final_answer", "answer": "..."}
      {"type": "unknown",      "raw": "..."}
    """
    text = response_text.strip()

    # Check for Final Answer first.
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {
            "type":   "final_answer",
            "answer": final_match.group(1).strip(),
        }

    # Check for Action + Action Input.
    action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    input_match  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)

    if action_match:
        tool_name  = action_match.group(1).strip().lower()
        tool_input = input_match.group(1).strip() if input_match else ""
        return {
            "type":  "action",
            "tool":  tool_name,
            "input": tool_input,
        }

    # Couldn't parse — return raw so we can display it.
    return {"type": "unknown", "raw": text}


# ─────────────────────────────────────────────
#  LLM CALL
#  Sends the current conversation to Groq
#  and returns the raw text response.
# ─────────────────────────────────────────────

def call_llm(messages: list[dict]) -> str:
    """
    Sends a list of messages to Groq and returns the assistant's reply.

    The 'messages' list is our entire conversation history:
      [system_prompt, user_question, assistant_thought, observation, ...]

    This is manually managed — there's no LangChain memory here.
    We build the list ourselves so you can see exactly what the model sees.
    """
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=messages,
        temperature=0.1,        # Low temperature for consistent tool use decisions
        max_tokens=512,
    )
    raw = response.choices[0].message.content
    # Qwen3 outputs <think>...</think> reasoning blocks — strip them
    # before parsing so they don't pollute the ReAct format detection.
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return clean


# ─────────────────────────────────────────────
#  RICH DISPLAY HELPERS
# ─────────────────────────────────────────────

def print_thought(thought_text: str, iteration: int) -> None:
    """Displays the agent's Thought step."""
    # Extract just the Thought line from the full response.
    thought_match = re.search(r"Thought:\s*(.+?)(?:\n|Action:|Final Answer:|$)",
                               thought_text, re.DOTALL | re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else thought_text

    console.print(
        Panel(
            f"[white]{thought}[/]",
            title=f"[yellow]🧠 THINK  [dim](iteration {iteration})[/][/]",
            border_style="yellow",
            padding=(0, 2),
        )
    )


def print_action(tool: str, tool_input: str) -> None:
    """Displays the ACT step — which tool + what input."""
    console.print(
        Panel(
            f"[bold cyan]Tool:[/]  [white]{tool}[/]\n"
            f"[bold cyan]Input:[/] [white]{tool_input}[/]",
            title="[cyan]⚡ ACT[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def print_observation(result: str) -> None:
    """Displays the OBSERVE step — what the tool returned."""
    console.print(
        Panel(
            f"[white]{result}[/]",
            title="[green]👁  OBSERVE[/]",
            border_style="green",
            padding=(0, 2),
        )
    )


def print_final_answer(answer: str) -> None:
    """Displays the agent's final answer."""
    console.print()
    console.print(
        Panel(
            f"[bold white]{answer}[/]",
            title="[bold green]✅ FINAL ANSWER[/]",
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )


def print_error(message: str) -> None:
    """Displays an error panel."""
    console.print(
        Panel(f"[red]{message}[/]", title="[red]ERROR[/]", border_style="red")
    )


# ─────────────────────────────────────────────
#  THE REACT LOOP
#  This is the heart of the lab.
#  Everything above was setup — this is the agent.
# ─────────────────────────────────────────────

def run_react_agent(user_question: str) -> None:
    """
    Runs the full ReAct loop for a single user question.

    HOW IT WORKS:
    1. Build the initial message list: [system, user_question]
    2. Call the LLM → it outputs Thought + Action + Action Input
    3. Parse the output to extract tool name and input
    4. Run the tool → get Observation
    5. Append the Observation to messages as a 'user' turn
       (we tell the model what the tool returned)
    6. Loop back to step 2
    7. Stop when LLM outputs "Final Answer:" OR we hit MAX_ITERATIONS

    This manual loop is exactly what LangChain's AgentExecutor
    and LangGraph's StateGraph automate. Now you know what's inside.
    """

    console.print()
    console.rule("[bold white]ReAct Agent Starting[/]")
    console.print(
        Panel(
            f"[bold white]{user_question}[/]",
            title="[bold yellow]❓ Question[/]",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()

    # ── Step 1: Initialise the conversation ───────────────────────────────────
    # The system prompt goes first. Then the user's question.
    # Every subsequent call will append to this list.
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_question},
    ]

    # ── Step 2: The loop ──────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERATIONS + 1):

        # ── THINK: Call the LLM ───────────────────────────────────────────────
        try:
            llm_response = call_llm(messages)
        except Exception as exc:
            print_error(f"LLM call failed: {exc}")
            return

        # Display the Thought portion of the response.
        print_thought(llm_response, iteration)

        # ── PARSE: Extract action or final answer ─────────────────────────────
        parsed = parse_llm_response(llm_response)

        # ── FINAL ANSWER: We're done ──────────────────────────────────────────
        if parsed["type"] == "final_answer":
            print_final_answer(parsed["answer"])
            console.print(
                f"\n[dim]Completed in {iteration} iteration(s). "
                f"Total messages in context: {len(messages) + 1}[/]\n"
            )
            return

        # ── ACT: Run the tool ─────────────────────────────────────────────────
        if parsed["type"] == "action":
            tool_name  = parsed["tool"]
            tool_input = parsed["input"]

            print_action(tool_name, tool_input)

            # Look up and call the tool.
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_input)
            else:
                observation = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(TOOLS.keys())}."
                )

            print_observation(observation)

            # ── OBSERVE: Feed result back into the conversation ───────────────
            # We append two messages:
            #   1. The assistant's Thought+Action (what it just said)
            #   2. A 'user' message with the Observation
            # This is how the agent "remembers" what tools returned.
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role":    "user",
                "content": f"Observation: {observation}\n\nContinue."
            })

            console.print()     # visual spacing between iterations
            continue

        # ── UNKNOWN: LLM didn't follow the format ─────────────────────────────
        if parsed["type"] == "unknown":
            # Sometimes the model skips the format. We nudge it back.
            console.print(
                Panel(
                    f"[dim]{parsed['raw'][:300]}[/]",
                    title="[yellow]⚠ Unexpected format — nudging agent[/]",
                    border_style="yellow",
                )
            )
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role":    "user",
                "content": (
                    "Please follow the format exactly:\n"
                    "Thought: ...\nAction: ...\nAction Input: ...\n"
                    "OR\nThought: ...\nFinal Answer: ..."
                ),
            })
            continue

    # ── MAX ITERATIONS REACHED ────────────────────────────────────────────────
    # This is the safety net. Real agents can loop forever without this.
    print_error(
        f"Agent stopped after {MAX_ITERATIONS} iterations without a Final Answer.\n"
        "This usually means the model got confused or the question is too complex.\n"
        "Try rephrasing your question."
    )


# ─────────────────────────────────────────────
#  EXAMPLE QUESTIONS
#  Pre-built questions that exercise each tool
#  and combinations of tools.
# ─────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    # Single tool — search
    "What is the ReAct pattern in AI?",
    # Single tool — calculate
    "What is 2 to the power of 16?",
    # Single tool — text analyze
    "Analyze this text: 'The quick brown fox jumps over the lazy dog near the river bank.'",
    # Multi-tool — search then calculate
    "What is LangChain and how many characters are in the word 'LangChain'?",
    # Multi-tool — search + reasoning
    "Explain what an LLM is and what company makes the Groq API?",
    # Edge case — topic not in knowledge base
    "What is the capital of France?",
]


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────

def print_welcome() -> None:
    console.print()
    console.print(
        Panel(
            "[bold white]CalderR Week 1 — Lab 1.2[/]\n"
            "[bold cyan]Manual ReAct Agent[/]  [dim]Pure Python · No Frameworks[/]\n\n"
            "[dim]This agent Reasons then Acts using tools, then loops until\n"
            "it has enough information to give you a Final Answer.[/]",
            border_style="blue",
            padding=(1, 4),
        )
    )


def print_examples() -> None:
    console.print("\n[bold yellow]Example Questions:[/]")
    for i, q in enumerate(EXAMPLE_QUESTIONS, start=1):
        console.print(f"  [cyan][{i}][/] {q}")
    console.print()


def main() -> None:
    print_welcome()

    while True:
        print_examples()

        console.print(
            "  [bold cyan][E][/] Enter example number (1–6)\n"
            "  [bold green][Q][/] Type your own question\n"
            "  [bold red][X][/] Exit\n"
        )

        choice = Prompt.ask("[bold white]Choose[/]").strip()

        if choice.lower() == "x":
            console.print("\n[dim]Goodbye.[/]\n")
            break

        elif choice.lower() == "q":
            question = Prompt.ask("[bold yellow]Your question[/]").strip()
            if question:
                run_react_agent(question)

        elif choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_QUESTIONS):
            question = EXAMPLE_QUESTIONS[int(choice) - 1]
            run_react_agent(question)

        else:
            console.print("[red]Invalid choice. Enter a number 1–6, Q, or X.[/]")

        console.rule("[dim]─[/]")


if __name__ == "__main__":
    main()