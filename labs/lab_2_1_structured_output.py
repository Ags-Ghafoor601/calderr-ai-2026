"""
CalderR Internship – Week 2, Lab 2.1
======================================
Structured Output Extractor — Job Posting Parser

WHAT THIS LAB BUILDS:
---------------------
A tool that takes unstructured, messy job posting text and extracts
clean, validated structured data using LangChain + Groq + Pydantic v2.

Every field is validated:
  • title           → string, required
  • company         → string, required
  • salary_min      → optional int (in USD)
  • salary_max      → optional int (in USD)
  • required_skills → list of strings
  • location        → string
  • remote_status   → enum: "remote" | "hybrid" | "onsite" | "unknown"

WHAT THIS TEACHES YOU:
----------------------
  • How Pydantic v2 models enforce output schemas on LLM responses
  • How `with_structured_output()` forces the LLM to return valid JSON
  • How to handle messy, ambiguous, real-world text extraction
  • How to build a validation report comparing expected vs actual output

ARCHITECTURE:
  Raw Job Posting Text (unstructured)
      ↓
  System Prompt + Extraction Instructions
      ↓
  ChatGroq.with_structured_output(JobPosting)
      ↓
  Pydantic Validation (automatic)
      ↓
  Rich Display + Validation Report

Run:
    python labs/lab_2_1_structured_output.py
"""

import os
import sys
import json
from enum import Enum
from typing import Optional
from datetime import datetime

# Fix Windows console encoding (cp1252 cannot handle Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
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
#  SECTION 1: Pydantic v2 Models
# ═══════════════════════════════════════════════

class RemoteStatus(str, Enum):
    """Enum for the remote work status of a job posting."""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class JobPosting(BaseModel):
    """
    Structured representation of a job posting.
    This Pydantic v2 model validates all fields and enforces types.
    """
    title: str = Field(
        description="The job title, e.g. 'Senior Python Developer'"
    )
    company: str = Field(
        description="The company or organization name"
    )
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum salary in USD (annual). None if not mentioned."
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum salary in USD (annual). None if not mentioned."
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="List of required technical skills mentioned in the posting"
    )
    location: str = Field(
        default="Not specified",
        description="Job location, e.g. 'San Francisco, CA' or 'Worldwide'"
    )
    remote_status: RemoteStatus = Field(
        default=RemoteStatus.UNKNOWN,
        description="Whether the job is remote, hybrid, onsite, or unknown"
    )

    @field_validator("title", "company")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        """Title and company must not be empty strings."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    @field_validator("required_skills")
    @classmethod
    def deduplicate_skills(cls, v: list[str]) -> list[str]:
        """Remove duplicate skills and empty strings."""
        seen = set()
        result = []
        for skill in v:
            s = skill.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)
        return result

    @model_validator(mode="after")
    def salary_range_check(self) -> "JobPosting":
        """If both salary fields are provided, min must be <= max."""
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_min > self.salary_max
        ):
            # Swap them rather than failing
            self.salary_min, self.salary_max = self.salary_max, self.salary_min
        return self


# ═══════════════════════════════════════════════
#  SECTION 2: Sample Job Postings (Unstructured)
# ═══════════════════════════════════════════════

SAMPLE_POSTINGS: list[dict] = [
    {
        "text": """
        🚀 We're hiring! Senior Backend Engineer at TechNova Inc.
        
        Come join our awesome team in Austin, Texas! We're looking for someone
        who knows Python inside and out, has experience with FastAPI or Django,
        and can work with PostgreSQL. Kubernetes experience is a huge plus.
        
        Compensation: $140,000 - $185,000/year + equity
        This is a hybrid role (3 days in office, 2 remote).
        
        Apply now at careers@technova.io
        """,
        "expected": {
            "title": "Senior Backend Engineer",
            "company": "TechNova Inc.",
            "salary_min": 140000,
            "salary_max": 185000,
            "required_skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Kubernetes"],
            "location": "Austin, Texas",
            "remote_status": "hybrid",
        },
    },
    {
        "text": """
        Job: ML Research Scientist
        Company: DeepMind
        Location: London, UK (fully remote available)
        
        We need someone with deep expertise in transformer architectures,
        PyTorch, and reinforcement learning. PhD preferred. Must have
        published at NeurIPS, ICML, or ICLR.
        
        Salary range not disclosed. Competitive compensation package.
        """,
        "expected": {
            "title": "ML Research Scientist",
            "company": "DeepMind",
            "salary_min": None,
            "salary_max": None,
            "required_skills": ["Transformer architectures", "PyTorch", "Reinforcement learning"],
            "location": "London, UK",
            "remote_status": "remote",
        },
    },
    {
        "text": """
        URGENT HIRE: full stack dev needed ASAP!!!
        
        small startup (CloudByte Solutions) looking for a full-stack developer.
        react, node.js, mongodb, AWS. must know docker.
        
        pay is between 90k and 120k. work from home forever, we dont have
        an office lol. based in new york but who cares
        """,
        "expected": {
            "title": "Full Stack Developer",
            "company": "CloudByte Solutions",
            "salary_min": 90000,
            "salary_max": 120000,
            "required_skills": ["React", "Node.js", "MongoDB", "AWS", "Docker"],
            "location": "New York",
            "remote_status": "remote",
        },
    },
    {
        "text": """
        Position Available: Junior Data Analyst
        Organization: World Health Organization (WHO)
        Duty Station: Geneva, Switzerland
        
        The successful candidate will support the data analytics team in
        processing health datasets using Excel, SQL, and Tableau. Basic
        Python scripting knowledge required. Must be fluent in English
        and French.
        
        Grade: P-2 (annual net salary approximately $57,000 - $73,000)
        This is an on-site position. No remote work.
        """,
        "expected": {
            "title": "Junior Data Analyst",
            "company": "World Health Organization (WHO)",
            "salary_min": 57000,
            "salary_max": 73000,
            "required_skills": ["Excel", "SQL", "Tableau", "Python"],
            "location": "Geneva, Switzerland",
            "remote_status": "onsite",
        },
    },
    {
        "text": """
        DevOps Engineer — Series B Startup

        Infra team @ Velocity Labs is growing! We need a DevOps/SRE
        to own our CI/CD pipelines, manage Terraform configs, and keep
        our k8s clusters humming. Experience with GitHub Actions,
        Prometheus, and Grafana required.

        💰 $130K-$170K + 0.2% equity
        📍 San Francisco or Remote (US only)
        🏠 Hybrid — 2 days in SF office
        """,
        "expected": {
            "title": "DevOps Engineer",
            "company": "Velocity Labs",
            "salary_min": 130000,
            "salary_max": 170000,
            "required_skills": ["CI/CD", "Terraform", "Kubernetes", "GitHub Actions", "Prometheus", "Grafana"],
            "location": "San Francisco",
            "remote_status": "hybrid",
        },
    },
]


# ═══════════════════════════════════════════════
#  SECTION 3: Extraction Engine
# ═══════════════════════════════════════════════

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precision data extraction engine. Your task is to extract
structured job posting information from unstructured text.

RULES:
1. Extract the exact job title as written (normalize capitalization).
2. Extract the company name exactly as written.
3. Extract salary as integers in USD. If salary is given in "K" notation
   (e.g., 90k), convert to full number (90000). If no salary, return null.
4. Extract ALL technical skills mentioned (tools, languages, frameworks).
5. Extract the location (city, state/country).
6. Determine remote status:
   - "remote" = fully remote / work from home
   - "hybrid" = mix of remote and in-office
   - "onsite" = must be in office / no remote
   - "unknown" = not mentioned or unclear

Be precise. Do not hallucinate information not present in the text.""",
    ),
    (
        "human",
        "Extract the job posting information from the following text:\n\n{job_text}",
    ),
])


def build_extractor() -> object:
    """Build the structured output extraction chain."""
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    structured_llm = llm.with_structured_output(JobPosting)
    chain = EXTRACTION_PROMPT | structured_llm
    return chain


def extract_job_posting(chain, text: str) -> JobPosting:
    """Extract structured data from a single job posting text."""
    result = chain.invoke({"job_text": text})
    return result


# ═══════════════════════════════════════════════
#  SECTION 4: Validation & Reporting
# ═══════════════════════════════════════════════

def compare_field(extracted, expected, field_name: str) -> tuple[bool, str]:
    """Compare a single field between extracted and expected values."""
    ext_val = getattr(extracted, field_name) if hasattr(extracted, field_name) else None
    exp_val = expected.get(field_name)

    # Handle enum comparison
    if isinstance(ext_val, RemoteStatus):
        ext_val = ext_val.value

    # Handle list comparison (case-insensitive)
    if isinstance(ext_val, list) and isinstance(exp_val, list):
        ext_set = {s.lower() for s in ext_val}
        exp_set = {s.lower() for s in exp_val}
        # Check if at least 60% of expected skills were found
        if len(exp_set) == 0:
            match = len(ext_set) == 0
        else:
            overlap = len(ext_set & exp_set)
            match = overlap / len(exp_set) >= 0.6
        detail = f"Found {len(ext_set & exp_set)}/{len(exp_set)} expected skills"
        return match, detail

    # Handle string comparison (case-insensitive, partial match)
    if isinstance(ext_val, str) and isinstance(exp_val, str):
        match = (
            ext_val.lower() == exp_val.lower()
            or exp_val.lower() in ext_val.lower()
            or ext_val.lower() in exp_val.lower()
        )
        return match, f"'{ext_val}' vs '{exp_val}'"

    # Handle numeric/None comparison
    if ext_val == exp_val:
        return True, f"{ext_val} == {exp_val}"

    # Handle numeric with tolerance (within 10%)
    if isinstance(ext_val, (int, float)) and isinstance(exp_val, (int, float)):
        if exp_val == 0:
            match = ext_val == 0
        else:
            match = abs(ext_val - exp_val) / abs(exp_val) <= 0.10
        return match, f"{ext_val} vs {exp_val}"

    return False, f"{ext_val} vs {exp_val}"


def generate_validation_report(
    results: list[tuple[int, JobPosting, dict]],
) -> None:
    """Generate a rich validation report comparing extracted vs expected."""
    console.print()
    console.print(Rule("📊 VALIDATION REPORT", style="bold cyan"))
    console.print()

    fields_to_check = [
        "title", "company", "salary_min", "salary_max",
        "required_skills", "location", "remote_status",
    ]

    total_checks = 0
    total_passed = 0

    for idx, extracted, expected in results:
        table = Table(
            title=f"Posting #{idx + 1}",
            box=box.ROUNDED,
            title_style="bold yellow",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Field", style="white", width=18)
        table.add_column("Expected", style="dim", width=30)
        table.add_column("Extracted", style="white", width=30)
        table.add_column("Match", justify="center", width=8)

        for field in fields_to_check:
            ext_val = getattr(extracted, field)
            exp_val = expected.get(field)
            passed, detail = compare_field(extracted, expected, field)
            total_checks += 1
            if passed:
                total_passed += 1

            # Format values for display
            if isinstance(ext_val, list):
                ext_display = ", ".join(ext_val) if ext_val else "[]"
            elif isinstance(ext_val, RemoteStatus):
                ext_display = ext_val.value
            else:
                ext_display = str(ext_val)

            if isinstance(exp_val, list):
                exp_display = ", ".join(exp_val) if exp_val else "[]"
            else:
                exp_display = str(exp_val)

            icon = "✅" if passed else "❌"
            table.add_row(field, exp_display, ext_display, icon)

        console.print(table)
        console.print()

    # Summary
    accuracy = (total_passed / total_checks * 100) if total_checks > 0 else 0
    color = "green" if accuracy >= 80 else "yellow" if accuracy >= 60 else "red"
    summary = Panel(
        Text.from_markup(
            f"[bold]Total Checks:[/] {total_checks}\n"
            f"[bold]Passed:[/] {total_passed}\n"
            f"[bold]Failed:[/] {total_checks - total_passed}\n"
            f"[bold {color}]Accuracy: {accuracy:.1f}%[/]"
        ),
        title="📈 Overall Accuracy",
        border_style=color,
        box=box.DOUBLE,
    )
    console.print(summary)


# ═══════════════════════════════════════════════
#  SECTION 5: Main Execution
# ═══════════════════════════════════════════════

def display_banner():
    """Display the lab banner."""
    banner = Text()
    banner.append("CalderR Internship - Week 2, Lab 2.1\n", style="bold cyan")
    banner.append("Structured Output Extractor\n", style="bold white")
    banner.append("Job Posting -> Pydantic Model -> Validated JSON\n", style="dim")
    banner.append(f"\nModel: {MODEL_NAME}  |  ", style="dim")
    banner.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style="dim")
    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))


def display_extracted_posting(idx: int, posting: JobPosting):
    """Display a single extracted posting in a rich panel."""
    salary_str = "Not disclosed"
    if posting.salary_min is not None and posting.salary_max is not None:
        salary_str = f"${posting.salary_min:,} - ${posting.salary_max:,}"
    elif posting.salary_min is not None:
        salary_str = f"${posting.salary_min:,}+"
    elif posting.salary_max is not None:
        salary_str = f"Up to ${posting.salary_max:,}"

    skills_str = ", ".join(posting.required_skills) if posting.required_skills else "None listed"

    content = Text()
    content.append(f"📌 Title:    ", style="bold")
    content.append(f"{posting.title}\n", style="white")
    content.append(f"🏢 Company:  ", style="bold")
    content.append(f"{posting.company}\n", style="white")
    content.append(f"💰 Salary:   ", style="bold")
    content.append(f"{salary_str}\n", style="green")
    content.append(f"🛠  Skills:   ", style="bold")
    content.append(f"{skills_str}\n", style="yellow")
    content.append(f"📍 Location: ", style="bold")
    content.append(f"{posting.location}\n", style="white")
    content.append(f"🏠 Remote:   ", style="bold")
    content.append(f"{posting.remote_status.value}", style="magenta")

    console.print(Panel(
        content,
        title=f"Extraction #{idx + 1}",
        border_style="green",
        box=box.ROUNDED,
    ))


def main():
    """Main entry point — extract all sample postings and generate report."""
    display_banner()

    console.print("\n[bold cyan]Building extraction chain...[/]")
    chain = build_extractor()
    console.print("[green]✓ Chain ready[/]\n")

    results: list[tuple[int, JobPosting, dict]] = []

    for idx, sample in enumerate(SAMPLE_POSTINGS):
        console.print(Rule(f"Processing Posting #{idx + 1}", style="dim"))

        # Show raw input (truncated)
        raw_preview = sample["text"].strip()[:120].replace("\n", " ")
        console.print(f"[dim]Raw input: \"{raw_preview}...\"[/]\n")

        try:
            posting = extract_job_posting(chain, sample["text"])
            display_extracted_posting(idx, posting)
            results.append((idx, posting, sample["expected"]))

            # Show raw JSON
            console.print(
                Panel(
                    json.dumps(posting.model_dump(mode="json"), indent=2),
                    title="Raw JSON Output",
                    border_style="dim",
                    box=box.SIMPLE,
                )
            )
        except Exception as e:
            console.print(f"[bold red]✗ Extraction failed: {e}[/]")
            console.print()

    # Generate validation report
    if results:
        generate_validation_report(results)

    console.print("\n[bold green]✓ Lab 2.1 complete![/]\n")


if __name__ == "__main__":
    main()
