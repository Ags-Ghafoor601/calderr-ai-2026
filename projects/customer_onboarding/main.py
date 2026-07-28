#!/usr/bin/env python3
"""
CalderR Internship – Week 4, Project 4-I-B
=============================================
Customer Onboarding Agent — Multi-Step Workflow with Human-in-the-Loop

WHAT THIS PROJECT BUILDS:
-------------------------
A complete customer onboarding workflow using LangGraph that:
  • Collects customer information (company, contact, size, revenue)
  • Validates the data (email format, required fields, business rules)
  • Determines account tier (Starter / Professional / Enterprise)
  • Routes Enterprise accounts for HUMAN APPROVAL (interrupt)
  • Auto-approves Starter and Professional accounts
  • Creates account with generated API key
  • Sends a welcome notification (simulated)
  • Schedules a follow-up meeting
  • Persists state in SQLite for reliable HITL workflows

ARCHITECTURE:
    ┌───────────────┐
    │  collect_info  │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │   validate     │
    └───────┬───────┘
            │
     ┌──────▼──────┐
     │  determine   │
     │    tier      │
     └──────┬──────┘
            │
    ┌───────▼────────┐
    │  needs human   │
    │  approval?     │
    └───┬────────┬───┘
   yes  │        │ no
  ┌─────▼────┐ ┌─▼──────────┐
  │  human   │ │   auto     │
  │  review  │ │  approve   │
  └─────┬────┘ └─┬──────────┘
   INTERRUPT│     │
  ┌─────▼────┐   │
  │  apply   │   │
  │ decision │   │
  └─────┬────┘   │
        └────┬───┘
    ┌────────▼────────┐
    │ create_account  │
    └────────┬────────┘
    ┌────────▼────────┐
    │ send_welcome    │
    └────────┬────────┘
    ┌────────▼──────────┐
    │schedule_followup  │
    └────────┬──────────┘
             │
            END

Run:
    python projects/customer_onboarding/main.py demo
    python projects/customer_onboarding/main.py onboard --name "Acme Corp" --contact "John" --email "john@acme.com" --size 50
    python projects/customer_onboarding/main.py pending
    python projects/customer_onboarding/main.py approve <thread-id>
    python projects/customer_onboarding/main.py reject <thread-id> --reason "Incomplete info"
    python projects/customer_onboarding/main.py graph
"""

import io
import os
import sys
import json
import time
import uuid
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Annotated
from operator import add

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.rule import Rule
from rich import box

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Add project root to path for imports
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(ROOT_DIR / ".env")

from models import (
    AccountTier, OnboardingStatus, CustomerInfo,
    AccountDetails, ValidationResult, SAMPLE_CUSTOMERS,
)

console = Console()
app = typer.Typer(
    name="customer-onboarding",
    help="🏢 Customer Onboarding Agent — Project 4-I-B",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHECKPOINT_DB = str(PROJECT_DIR / ".onboarding_checkpoint.db")
ONBOARDING_LOG = PROJECT_DIR / "onboarding_log.json"

# Enterprise threshold: companies with > 100 employees or > $1M revenue
ENTERPRISE_SIZE_THRESHOLD = 100
ENTERPRISE_REVENUE_THRESHOLD = 1_000_000


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

def get_llm():
    """Create a ChatGroq LLM instance."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=GROQ_API_KEY,
    )


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class OnboardingState(TypedDict):
    """State for the customer onboarding workflow.

    Fields are grouped by workflow stage:
      INPUT: Raw customer data
      VALIDATION: Results of data validation
      TIER: Account classification
      APPROVAL: Human-in-the-loop approval tracking
      ACCOUNT: Generated account details
      NOTIFICATION: Welcome and follow-up tracking
      META: Processing logs and status
    """
    # INPUT
    company_name: str
    contact_name: str
    contact_email: str
    phone: str
    company_size: int
    annual_revenue: float
    industry: str
    use_case: str

    # VALIDATION
    is_valid: bool
    validation_errors: list[str]
    validation_warnings: list[str]

    # TIER
    account_tier: str
    requires_human_approval: bool
    approval_reason: str

    # APPROVAL
    human_decision: str          # "approve" or "reject"
    human_notes: str
    awaiting_human: bool

    # ACCOUNT
    account_id: str
    api_key: str
    account_created: bool

    # NOTIFICATION
    welcome_sent: bool
    followup_date: str

    # META
    thread_id: str
    status: str
    processing_log: Annotated[list[str], add]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def collect_info(state: OnboardingState) -> dict:
    """Register customer information in the pipeline."""
    return {
        "status": OnboardingStatus.INFO_COLLECTED.value,
        "processing_log": [
            f"[COLLECT] 📋 Customer info received — "
            f"Company: {state['company_name']}, "
            f"Contact: {state['contact_name']} ({state['contact_email']}), "
            f"Size: {state['company_size']} employees, "
            f"Revenue: ${state['annual_revenue']:,.0f}"
        ],
    }


def validate_info(state: OnboardingState) -> dict:
    """Validate customer information against business rules."""
    errors = []
    warnings = []

    # Required fields
    if not state.get("company_name", "").strip():
        errors.append("Company name is required")
    if not state.get("contact_name", "").strip():
        errors.append("Contact name is required")
    if not state.get("contact_email", "").strip():
        errors.append("Contact email is required")

    # Email format
    email = state.get("contact_email", "")
    if email and "@" not in email:
        errors.append(f"Invalid email format: {email}")

    # Company size
    size = state.get("company_size", 0)
    if size < 1:
        errors.append(f"Company size must be at least 1 (got {size})")

    # Revenue
    revenue = state.get("annual_revenue", 0)
    if revenue < 0:
        errors.append(f"Annual revenue cannot be negative (got ${revenue:,.0f})")

    # Warnings
    if not state.get("phone", "").strip():
        warnings.append("No phone number provided — follow-up may be limited")
    if size > 1000:
        warnings.append(f"Very large company ({size:,} employees) — may need custom integration")

    is_valid = len(errors) == 0
    status = "✅ PASSED" if is_valid else f"❌ FAILED ({len(errors)} errors)"

    return {
        "is_valid": is_valid,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "status": OnboardingStatus.VALIDATED.value if is_valid else "rejected",
        "processing_log": [
            f"[VALIDATE] {status} — "
            f"{len(errors)} errors, {len(warnings)} warnings"
        ],
    }


def determine_tier(state: OnboardingState) -> dict:
    """Classify the account tier based on company size and revenue."""
    size = state.get("company_size", 0)
    revenue = state.get("annual_revenue", 0)

    if size > ENTERPRISE_SIZE_THRESHOLD or revenue > ENTERPRISE_REVENUE_THRESHOLD:
        tier = AccountTier.ENTERPRISE.value
        requires_approval = True
        reason = (
            f"Enterprise account (size: {size:,}, revenue: ${revenue:,.0f}) "
            f"exceeds thresholds (>{ENTERPRISE_SIZE_THRESHOLD} employees or "
            f">${ENTERPRISE_REVENUE_THRESHOLD:,} revenue)"
        )
    elif size > 10 or revenue > 100_000:
        tier = AccountTier.PROFESSIONAL.value
        requires_approval = False
        reason = "Professional tier — auto-approved"
    else:
        tier = AccountTier.STARTER.value
        requires_approval = False
        reason = "Starter tier — auto-approved"

    emoji = {"enterprise": "🏢", "professional": "💼", "starter": "🌱"}[tier]
    return {
        "account_tier": tier,
        "requires_human_approval": requires_approval,
        "approval_reason": reason,
        "processing_log": [
            f"[TIER] {emoji} {tier.upper()} — {reason}"
        ],
    }


def auto_approve(state: OnboardingState) -> dict:
    """Automatically approve non-enterprise accounts."""
    return {
        "human_decision": "approve",
        "human_notes": f"Auto-approved: {state['account_tier']} tier",
        "status": OnboardingStatus.APPROVED.value,
        "processing_log": [
            f"[AUTO-APPROVE] ✅ {state['account_tier'].title()} account "
            f"auto-approved for {state['company_name']}"
        ],
    }


def human_review(state: OnboardingState) -> dict:
    """Flag for human review — enterprise accounts need manual approval."""
    return {
        "awaiting_human": True,
        "status": OnboardingStatus.AWAITING_APPROVAL.value,
        "processing_log": [
            f"[HUMAN-REVIEW] ⏸️  Enterprise account for {state['company_name']} "
            f"queued for human approval — {state['approval_reason']}"
        ],
    }


def apply_human_decision(state: OnboardingState) -> dict:
    """Apply the human reviewer's approval or rejection."""
    decision = state.get("human_decision", "reject").lower()
    is_approved = decision in ("approve", "approved")

    return {
        "awaiting_human": False,
        "status": OnboardingStatus.APPROVED.value if is_approved else OnboardingStatus.REJECTED.value,
        "processing_log": [
            f"[HUMAN-DECISION] {'✅ APPROVED' if is_approved else '❌ REJECTED'} — "
            f"{state['company_name']} by human reviewer: {state.get('human_notes', 'No notes')}"
        ],
    }


def create_account(state: OnboardingState) -> dict:
    """Generate account credentials and register the new customer."""
    # Generate unique account ID and API key
    seed = f"{state['company_name']}-{state['contact_email']}-{time.time()}"
    account_id = "ACC-" + hashlib.sha256(seed.encode()).hexdigest()[:8].upper()
    api_key = "sk-" + secrets.token_hex(24)

    return {
        "account_id": account_id,
        "api_key": api_key,
        "account_created": True,
        "status": OnboardingStatus.ACCOUNT_CREATED.value,
        "processing_log": [
            f"[ACCOUNT] 🔑 Account created — ID: {account_id}, "
            f"Tier: {state['account_tier']}, API key generated"
        ],
    }


def send_welcome(state: OnboardingState) -> dict:
    """Send welcome notification to the new customer (simulated)."""
    llm = get_llm()

    try:
        result = llm.invoke([
            SystemMessage(content="You write brief, professional welcome emails. Keep it under 100 words."),
            HumanMessage(content=(
                f"Write a welcome email for {state['contact_name']} from {state['company_name']}. "
                f"They signed up for a {state['account_tier']} account. "
                f"Their account ID is {state['account_id']}. "
                f"Their use case is: {state['use_case']}."
            )),
        ])
        welcome_msg = result.content.strip()
    except Exception:
        welcome_msg = (
            f"Welcome, {state['contact_name']}! Your {state['account_tier']} account "
            f"({state['account_id']}) is ready. Get started at docs.example.com."
        )

    return {
        "welcome_sent": True,
        "status": OnboardingStatus.NOTIFIED.value,
        "processing_log": [
            f"[WELCOME] 📧 Welcome email sent to {state['contact_email']} — "
            f"({len(welcome_msg)} chars)"
        ],
    }


def schedule_followup(state: OnboardingState) -> dict:
    """Schedule a follow-up meeting based on account tier."""
    tier = state.get("account_tier", "starter")

    # Enterprise: 2 days, Professional: 5 days, Starter: 7 days
    days_map = {"enterprise": 2, "professional": 5, "starter": 7}
    days = days_map.get(tier, 7)
    followup = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

    return {
        "followup_date": followup,
        "status": OnboardingStatus.COMPLETED.value,
        "processing_log": [
            f"[FOLLOWUP] 📅 Follow-up scheduled for {followup} "
            f"({days} days — {tier} tier schedule)"
        ],
    }


def handle_invalid(state: OnboardingState) -> dict:
    """Handle rejected applications due to validation errors."""
    errors = state.get("validation_errors", [])
    return {
        "status": OnboardingStatus.REJECTED.value,
        "processing_log": [
            f"[REJECTED] ❌ Onboarding rejected — {len(errors)} validation error(s): "
            + "; ".join(errors)
        ],
    }


def handle_rejected(state: OnboardingState) -> dict:
    """Handle applications rejected by human reviewer."""
    return {
        "status": OnboardingStatus.REJECTED.value,
        "processing_log": [
            f"[REJECTED] ❌ Application rejected by human reviewer — "
            f"Company: {state['company_name']}, Notes: {state.get('human_notes', 'None')}"
        ],
    }


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_validation(state: OnboardingState) -> str:
    """Route based on validation result."""
    if state.get("is_valid", False):
        return "determine_tier"
    return "handle_invalid"


def route_after_tier(state: OnboardingState) -> str:
    """Route based on account tier — enterprise needs human approval."""
    if state.get("requires_human_approval", False):
        return "human_review"
    return "auto_approve"


def route_after_human_decision(state: OnboardingState) -> str:
    """Route based on human reviewer's decision."""
    decision = state.get("human_decision", "reject").lower()
    if decision in ("approve", "approved"):
        return "create_account"
    return "handle_rejected"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the customer onboarding graph."""
    graph = StateGraph(OnboardingState)

    # Add all nodes
    graph.add_node("collect_info", collect_info)
    graph.add_node("validate_info", validate_info)
    graph.add_node("determine_tier", determine_tier)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_human_decision", apply_human_decision)
    graph.add_node("create_account", create_account)
    graph.add_node("send_welcome", send_welcome)
    graph.add_node("schedule_followup", schedule_followup)
    graph.add_node("handle_invalid", handle_invalid)
    graph.add_node("handle_rejected", handle_rejected)

    # Entry point
    graph.set_entry_point("collect_info")

    # Edges
    graph.add_edge("collect_info", "validate_info")

    # Conditional: valid → determine tier, invalid → reject
    graph.add_conditional_edges(
        "validate_info",
        route_after_validation,
        {
            "determine_tier": "determine_tier",
            "handle_invalid": "handle_invalid",
        },
    )

    # Conditional: enterprise → human review, otherwise → auto approve
    graph.add_conditional_edges(
        "determine_tier",
        route_after_tier,
        {
            "human_review": "human_review",
            "auto_approve": "auto_approve",
        },
    )

    # Human review path
    graph.add_edge("human_review", "apply_human_decision")
    graph.add_conditional_edges(
        "apply_human_decision",
        route_after_human_decision,
        {
            "create_account": "create_account",
            "handle_rejected": "handle_rejected",
        },
    )

    # Auto-approve goes to create account
    graph.add_edge("auto_approve", "create_account")

    # Account creation → welcome → followup → END
    graph.add_edge("create_account", "send_welcome")
    graph.add_edge("send_welcome", "schedule_followup")
    graph.add_edge("schedule_followup", END)

    # Terminal nodes
    graph.add_edge("handle_invalid", END)
    graph.add_edge("handle_rejected", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_human_decision"],
    )


def get_initial_state(customer: dict, thread_id: str = "") -> dict:
    """Create initial state from customer data."""
    return {
        "company_name": customer.get("company_name", ""),
        "contact_name": customer.get("contact_name", ""),
        "contact_email": customer.get("contact_email", ""),
        "phone": customer.get("phone", ""),
        "company_size": customer.get("company_size", 0),
        "annual_revenue": customer.get("annual_revenue", 0.0),
        "industry": customer.get("industry", ""),
        "use_case": customer.get("use_case", ""),
        "is_valid": False,
        "validation_errors": [],
        "validation_warnings": [],
        "account_tier": "",
        "requires_human_approval": False,
        "approval_reason": "",
        "human_decision": "",
        "human_notes": "",
        "awaiting_human": False,
        "account_id": "",
        "api_key": "",
        "account_created": False,
        "welcome_sent": False,
        "followup_date": "",
        "thread_id": thread_id or str(uuid.uuid4())[:8],
        "status": "pending",
        "processing_log": [],
    }


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def display_result(result: dict, title: str = "Onboarding Result"):
    """Display onboarding result with rich formatting."""
    console.print()

    # Processing timeline
    log = result.get("processing_log", [])
    if log:
        tree = Tree("📋 [bold]Onboarding Timeline[/]")
        for entry in log:
            if "✅" in entry or "APPROVE" in entry or "ACCOUNT" in entry or "WELCOME" in entry or "FOLLOWUP" in entry:
                tree.add(f"[green]{entry}[/]")
            elif "❌" in entry or "REJECTED" in entry:
                tree.add(f"[red]{entry}[/]")
            elif "⏸️" in entry or "HUMAN" in entry:
                tree.add(f"[yellow]{entry}[/]")
            elif "TIER" in entry:
                tree.add(f"[magenta]{entry}[/]")
            else:
                tree.add(f"[cyan]{entry}[/]")
        console.print(tree)

    # Status
    status = result.get("status", "unknown")
    if status == OnboardingStatus.COMPLETED.value:
        console.print()
        table = Table(title="🎉 Account Summary", box=box.DOUBLE)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="bold white")
        table.add_row("Company", result.get("company_name", ""))
        table.add_row("Contact", f"{result.get('contact_name', '')} ({result.get('contact_email', '')})")
        table.add_row("Account ID", f"[bold green]{result.get('account_id', '')}[/]")
        table.add_row("Tier", f"[bold]{result.get('account_tier', '').upper()}[/]")
        table.add_row("API Key", f"[dim]{result.get('api_key', '')[:20]}...{'*' * 10}[/]")
        table.add_row("Welcome Sent", "✅ Yes" if result.get("welcome_sent") else "❌ No")
        table.add_row("Follow-up Date", result.get("followup_date", "N/A"))
        table.add_row("Status", f"[bold green]{status.upper()}[/]")
        console.print(table)

    elif result.get("awaiting_human"):
        console.print()
        console.print(Panel(
            f"[bold yellow]⏸️ AWAITING HUMAN APPROVAL[/]\n\n"
            f"[cyan]Company:[/] {result.get('company_name', '')}\n"
            f"[cyan]Contact:[/] {result.get('contact_name', '')} ({result.get('contact_email', '')})\n"
            f"[cyan]Size:[/] {result.get('company_size', 0):,} employees\n"
            f"[cyan]Revenue:[/] ${result.get('annual_revenue', 0):,.0f}\n"
            f"[cyan]Tier:[/] {result.get('account_tier', '').upper()}\n"
            f"[cyan]Reason:[/] {result.get('approval_reason', '')}",
            title=f"🏢 {title}",
            border_style="yellow",
        ))

    elif status == OnboardingStatus.REJECTED.value:
        console.print()
        errors = result.get("validation_errors", [])
        error_text = "\n".join(f"  ⚠️  {e}" for e in errors) if errors else "  Rejected by reviewer"
        console.print(Panel(
            f"[bold red]❌ APPLICATION REJECTED[/]\n\n"
            f"{error_text}\n\n"
            f"[cyan]Company:[/] {result.get('company_name', 'N/A')}\n"
            f"[cyan]Status:[/] {status}",
            title=f"🛑 {title}",
            border_style="red",
        ))


def display_graph_structure():
    """Display the onboarding graph structure."""
    console.print()
    tree = Tree("🔷 [bold cyan]Customer Onboarding Graph[/]")

    collect = tree.add("📋 [bold]collect_info[/] — Register customer data")
    validate = collect.add("🔍 [bold]validate_info[/] — Check fields & business rules")
    cond1 = validate.add("⚡ [bold yellow]CONDITIONAL: valid?[/]")

    invalid = cond1.add("❌ [red]invalid[/] → [bold]handle_invalid[/] → END")

    tier = cond1.add("✅ [green]valid[/] → [bold]determine_tier[/] — Classify account")
    cond2 = tier.add("⚡ [bold yellow]CONDITIONAL: enterprise?[/]")

    auto = cond2.add("🌱 [green]starter/professional[/] → [bold]auto_approve[/]")

    ent = cond2.add("🏢 [yellow]enterprise[/] → [bold]human_review[/]")
    interrupt = ent.add("⏸️  [bold red]INTERRUPT[/] (await human approval)")
    apply = interrupt.add("▶️  [bold]apply_human_decision[/]")
    cond3 = apply.add("⚡ [bold yellow]CONDITIONAL: approved?[/]")
    cond3.add("❌ [red]rejected[/] → [bold]handle_rejected[/] → END")
    cond3.add("✅ [green]approved[/] ↓")

    auto.add("↓ (merge)")

    create = tree.add("🔑 [bold]create_account[/] — Generate ID & API key")
    welcome = create.add("📧 [bold]send_welcome[/] — LLM-generated welcome email")
    followup = welcome.add("📅 [bold]schedule_followup[/] — Schedule meeting")
    followup.add("✅ [bold green]END[/]")

    console.print(Panel(tree, title="📐 Graph Architecture", border_style="blue"))

    config = Table(title="⚙️  Configuration", box=box.ROUNDED)
    config.add_column("Parameter", style="bold cyan")
    config.add_column("Value", style="bold white")
    config.add_row("Enterprise Threshold (Size)", f">{ENTERPRISE_SIZE_THRESHOLD} employees")
    config.add_row("Enterprise Threshold (Revenue)", f">${ENTERPRISE_REVENUE_THRESHOLD:,}")
    config.add_row("LLM Model", "llama-3.1-8b-instant")
    config.add_row("Persistence", "SqliteSaver (SQLite)")
    config.add_row("Interrupt Point", "Before apply_human_decision")
    console.print(config)


def save_to_log(result: dict):
    """Save onboarding result to log file."""
    entry = {
        "company_name": result.get("company_name", ""),
        "contact_email": result.get("contact_email", ""),
        "account_tier": result.get("account_tier", ""),
        "account_id": result.get("account_id", ""),
        "status": result.get("status", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    existing = []
    if ONBOARDING_LOG.exists():
        try:
            existing = json.loads(ONBOARDING_LOG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            existing = []

    existing.append(entry)
    ONBOARDING_LOG.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def onboard(
    name: str = typer.Option(..., "--name", help="Company name"),
    contact: str = typer.Option(..., "--contact", help="Contact person name"),
    email: str = typer.Option(..., "--email", help="Contact email"),
    size: int = typer.Option(..., "--size", help="Number of employees"),
    revenue: float = typer.Option(0, "--revenue", help="Annual revenue (USD)"),
    industry: str = typer.Option("Technology", "--industry", help="Industry"),
    use_case: str = typer.Option("General", "--use-case", help="Use case description"),
    phone: str = typer.Option("", "--phone", help="Phone number"),
):
    """Onboard a new customer."""
    console.print(Rule("🏢 Customer Onboarding — Project 4-I-B", style="bold blue"))
    display_graph_structure()
    console.print()
    console.print(Rule("🚀 Processing Onboarding", style="bold green"))

    customer = {
        "company_name": name,
        "contact_name": contact,
        "contact_email": email,
        "phone": phone,
        "company_size": size,
        "annual_revenue": revenue,
        "industry": industry,
        "use_case": use_case,
    }

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)
        thread_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": thread_id}}

        result = compiled.invoke(get_initial_state(customer, thread_id), config)
        display_result(result)
        save_to_log(result)

        if result.get("awaiting_human"):
            console.print()
            console.print(Panel(
                f"To approve:  [bold cyan]python projects/customer_onboarding/main.py approve {thread_id}[/]\n"
                f"To reject:   [bold cyan]python projects/customer_onboarding/main.py reject {thread_id}[/]",
                title="📋 Next Steps",
                border_style="yellow",
            ))


@app.command(name="approve")
def approve_cmd(
    thread_id: str = typer.Argument(..., help="Thread ID of the pending onboarding"),
    notes: str = typer.Option("Approved by manager", "--notes", "-n", help="Approval notes"),
):
    """Approve a pending enterprise onboarding."""
    console.print(Rule(f"✅ Approving Thread {thread_id}", style="bold green"))

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        current = compiled.get_state(config)
        if not current or not current.values:
            console.print(f"[red]❌ No pending onboarding found for thread {thread_id}[/]")
            raise typer.Exit(1)

        compiled.update_state(config, {
            "human_decision": "approve",
            "human_notes": notes,
        })

        result = compiled.invoke(None, config)
        display_result(result, "Approval Complete")
        save_to_log(result)


@app.command(name="reject")
def reject_cmd(
    thread_id: str = typer.Argument(..., help="Thread ID of the pending onboarding"),
    reason: str = typer.Option("Application does not meet requirements", "--reason", "-r"),
):
    """Reject a pending enterprise onboarding."""
    console.print(Rule(f"❌ Rejecting Thread {thread_id}", style="bold red"))

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        current = compiled.get_state(config)
        if not current or not current.values:
            console.print(f"[red]❌ No pending onboarding found for thread {thread_id}[/]")
            raise typer.Exit(1)

        compiled.update_state(config, {
            "human_decision": "reject",
            "human_notes": reason,
        })

        result = compiled.invoke(None, config)
        display_result(result, "Rejection Complete")
        save_to_log(result)


@app.command(name="graph")
def graph_cmd():
    """Display the onboarding graph structure."""
    console.print(Rule("📐 Customer Onboarding Graph — Project 4-I-B", style="bold blue"))
    display_graph_structure()


@app.command()
def demo():
    """Run a full demonstration with multiple customer types."""
    console.print(Panel(
        "[bold cyan]Project 4-I-B — Customer Onboarding Agent[/]\n"
        "[dim]Multi-step LangGraph workflow with human-in-the-loop for enterprise accounts[/]\n\n"
        "This demo onboards 5 customers:\n"
        "  1️⃣  [green]TechStart Inc.[/] — Small (5 employees) → Starter, auto-approved\n"
        "  2️⃣  [yellow]MegaCorp Global[/] — Large (500 employees, $5M) → Enterprise, human approval\n"
        "  3️⃣  [green]GreenLeaf Analytics[/] — Medium (45 employees) → Professional, auto-approved\n"
        "  4️⃣  [yellow]DataFlow Systems[/] — Medium-large (120 employees) → Enterprise, human approval\n"
        "  5️⃣  [red]Invalid Application[/] — Missing/invalid fields → Rejected",
        title="🔬 Demo Mode",
        border_style="blue",
        padding=(1, 2),
    ))

    display_graph_structure()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)

        labels = [
            ("TechStart Inc. — Starter (auto-approve)", "green"),
            ("MegaCorp Global — Enterprise (human approval)", "yellow"),
            ("GreenLeaf Analytics — Professional (auto-approve)", "green"),
            ("DataFlow Systems — Enterprise (human approval)", "yellow"),
            ("Invalid Application — Validation Rejection", "red"),
        ]

        for i, (customer, (label, color)) in enumerate(zip(SAMPLE_CUSTOMERS, labels), 1):
            console.print()
            console.print(Rule(f"🧪 Test {i}/{len(SAMPLE_CUSTOMERS)}: {label}", style=f"bold {color}"))

            thread_id = f"demo-{i}-{str(uuid.uuid4())[:4]}"
            config = {"configurable": {"thread_id": thread_id}}
            initial = get_initial_state(customer, thread_id)

            result = compiled.invoke(initial, config)

            # If awaiting human approval, simulate approval
            if result.get("awaiting_human"):
                console.print()
                console.print(Panel(
                    "[bold yellow]⏸️ INTERRUPTED — Enterprise account needs human approval[/]\n"
                    "[dim]State persisted to SQLite checkpoint.[/]\n"
                    "[bold green]▶️ Simulating manager: APPROVED[/]",
                    border_style="yellow",
                ))

                compiled.update_state(config, {
                    "human_decision": "approve",
                    "human_notes": "Approved by demo manager — enterprise account verified",
                })

                result = compiled.invoke(None, config)

            display_result(result)
            save_to_log(result)

    # Summary
    console.print()
    console.print(Panel(
        "[bold green]✅ Demo complete![/]\n\n"
        "Key features demonstrated:\n"
        "  • [cyan]Conditional routing[/] based on account tier\n"
        "  • [cyan]Human-in-the-loop[/] for enterprise accounts\n"
        "  • [cyan]SqliteSaver persistence[/] across interrupts\n"
        "  • [cyan]Data validation[/] with detailed error reporting\n"
        "  • [cyan]LLM-generated[/] welcome emails\n"
        "  • [cyan]Scheduled follow-ups[/] based on tier\n"
        "  • [cyan]Audit logging[/] for all onboarding decisions",
        title="📊 Demo Summary",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
