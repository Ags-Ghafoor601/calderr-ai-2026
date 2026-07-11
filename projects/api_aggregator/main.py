"""
CalderR Internship – Week 2, Intermediate Project
=====================================================
API Aggregator Agent — Morning Briefing Generator

WHAT THIS PROJECT BUILDS:
-------------------------
An agent that pulls data from 3 public APIs in parallel and
synthesizes a morning briefing report:
  • Weather (wttr.in — no API key)
  • Tech News (Hacker News API — no API key)
  • Crypto Market (CoinGecko — no API key)

The raw data is then fed to Groq LLM for AI synthesis into
a cohesive, well-written morning briefing.

OUTPUT:
  • HTML report (viewable in browser)
  • Markdown report (viewable in terminal / GitHub)
  • Raw JSON data (for debugging)

ARCHITECTURE:
  Parallel Tool Calls (asyncio.gather)
    ├── Weather API (wttr.in)
    ├── News API (Hacker News)
    └── Finance API (CoinGecko)
      ↓
  Data Aggregator (merge results)
      ↓
  LLM Synthesizer (Groq + LangChain)
      ↓
  Report Generator (HTML + Markdown)

Run:
    python projects/api_aggregator/main.py
    python projects/api_aggregator/main.py --city London
    python projects/api_aggregator/main.py --city Tokyo --no-html
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding (cp1252 cannot handle Unicode)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also add the overall project root for .env
REPO_ROOT = PROJECT_ROOT.parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent import gather_all_data, synthesize_report
from report_generator import generate_html_report, generate_markdown_report

console = Console(force_terminal=True)


def display_banner(city: str):
    """Display the project banner."""
    banner = Text()
    banner.append("CalderR Internship — Week 2\n", style="bold cyan")
    banner.append("API Aggregator Agent\n", style="bold white")
    banner.append("Morning Briefing Generator\n\n", style="dim")
    banner.append("📡 Data Sources:\n", style="bold")
    banner.append("  • 🌤️  Weather    → wttr.in (no API key)\n", style="dim")
    banner.append("  • 📰  News       → Hacker News API (no API key)\n", style="dim")
    banner.append("  • 💰  Finance    → CoinGecko API (no API key)\n", style="dim")
    banner.append(f"\n📍 City: {city}\n", style="yellow")
    banner.append(f"📅 Date: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}", style="dim")
    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))


async def run_pipeline(city: str, save_html: bool = True):
    """Run the full aggregation pipeline."""

    # Step 1: Gather data from all APIs in parallel
    console.print("\n[bold cyan]Step 1: Fetching data from 3 APIs in parallel...[/]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching weather, news, and finance data...", total=None)
        raw_data = await gather_all_data(city)
        progress.remove_task(task)

    # Display fetch results
    weather_status = raw_data["weather"].get("status", "unknown")
    news_count = len(raw_data["news"].get("stories", []))
    finance_count = len(raw_data["finance"].get("coins", []))

    console.print(f"  🌤️  Weather:  {'[green]✓[/]' if weather_status == 'success' else '[red]✗[/]'} ({raw_data['weather'].get('city', city)})")
    console.print(f"  📰  News:     [green]✓[/] ({news_count} stories fetched)")
    console.print(f"  💰  Finance:  [green]✓[/] ({finance_count} coins tracked)")
    console.print()

    # Step 2: AI Synthesis
    console.print("[bold cyan]Step 2: AI Synthesis via Groq LLM...[/]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesizing morning briefing...", total=None)
        report_text = synthesize_report(raw_data)
        progress.remove_task(task)

    console.print("  🤖  Synthesis: [green]✓[/] Report generated")
    console.print()

    # Step 3: Generate reports
    console.print("[bold cyan]Step 3: Generating report files...[/]")

    reports_dir = PROJECT_ROOT / "sample_reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Markdown report
    md_report = generate_markdown_report(report_text, raw_data, city)
    md_path = reports_dir / f"briefing_{timestamp}.md"
    md_path.write_text(md_report, encoding="utf-8")
    console.print(f"  📝  Markdown:  [green]✓[/] → {md_path}")

    # Save HTML report
    if save_html:
        html_report = generate_html_report(report_text, raw_data, city)
        html_path = reports_dir / f"briefing_{timestamp}.html"
        html_path.write_text(html_report, encoding="utf-8")
        console.print(f"  🌐  HTML:      [green]✓[/] → {html_path}")

    # Save raw JSON
    json_path = reports_dir / f"briefing_{timestamp}_raw.json"
    json_path.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")
    console.print(f"  📊  Raw JSON:  [green]✓[/] → {json_path}")

    console.print()

    # Step 4: Display the report in terminal
    console.print(Rule("📋 Morning Briefing", style="bold cyan"))
    console.print()
    console.print(Markdown(md_report))

    # Summary
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold green]✓ Morning briefing generated successfully![/]\n\n"
            f"[bold]Files saved to:[/] {reports_dir}\n"
            f"[bold]APIs called:[/] 3 (weather, news, finance)\n"
            f"[bold]Data points:[/] {1 + news_count + finance_count}\n"
            f"[bold]LLM model:[/] llama-3.1-8b-instant via Groq"
        ),
        title="✅ Complete",
        border_style="green",
        box=box.DOUBLE,
    ))


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="API Aggregator Agent — Morning Briefing Generator"
    )
    parser.add_argument(
        "--city", "-c",
        type=str,
        default="New York",
        help="City for weather data (default: New York)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    args = parser.parse_args()

    display_banner(args.city)
    asyncio.run(run_pipeline(args.city, save_html=not args.no_html))


if __name__ == "__main__":
    main()
