"""
API Aggregator Agent — LangChain Agent with Parallel Tool Calling
===================================================================
Orchestrates 3 data-source tools (weather, news, finance), fetches
data in parallel via asyncio, and uses Groq LLM to synthesize a
cohesive morning briefing report.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

from tenacity import retry, wait_exponential, stop_after_attempt

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.weather_tool import fetch_weather
from tools.news_tool import fetch_news
from tools.finance_tool import fetch_finance


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional morning briefing analyst. Your job is to take
raw data from three sources (weather, news, finance) and compose a concise,
engaging, and well-structured morning briefing report.

STYLE GUIDELINES:
- Write in a professional but friendly tone, like a premium newsletter
- Use clear section headers with emojis
- Highlight the most important/interesting items
- Include specific numbers, temperatures, and percentages
- End with a brief "outlook" or takeaway
- Keep it concise — aim for 300-400 words total

FORMAT: Write in clean Markdown with headers, bullet points, and bold text.""",
    ),
    (
        "human",
        """Compose a morning briefing report from the following raw data:

WEATHER DATA:
{weather_data}

NEWS HEADLINES:
{news_data}

FINANCE/CRYPTO MARKET DATA:
{finance_data}

Today's date: {current_date}

Write the briefing now.""",
    ),
])


async def gather_all_data(city: str = "New York") -> dict:
    """
    Fetch data from all 3 APIs in parallel using asyncio.gather().

    Args:
        city: City for weather data

    Returns:
        Dictionary with weather, news, and finance data.
    """
    # Run all 3 API calls concurrently
    weather_task = fetch_weather(city)
    news_task = fetch_news(num_stories=10)
    finance_task = fetch_finance()

    weather_data, news_data, finance_data = await asyncio.gather(
        weather_task,
        news_task,
        finance_task,
        return_exceptions=True,
    )

    # Handle any exceptions that were returned instead of raising
    if isinstance(weather_data, Exception):
        weather_data = {"status": "error", "error": str(weather_data), "city": city}
    if isinstance(news_data, Exception):
        news_data = {"status": "error", "error": str(news_data), "stories": []}
    if isinstance(finance_data, Exception):
        finance_data = {"status": "error", "error": str(finance_data), "coins": []}

    return {
        "weather": weather_data,
        "news": news_data,
        "finance": finance_data,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(5))
def synthesize_report(raw_data: dict) -> str:
    """
    Use Groq LLM to synthesize raw data into a morning briefing report.

    Args:
        raw_data: Dictionary containing weather, news, and finance data

    Returns:
        Formatted morning briefing as Markdown string
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set. Cannot generate AI synthesis."

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.7,  # Slightly creative for engaging writing
        api_key=GROQ_API_KEY,
        max_tokens=1024,
    )

    chain = SYNTHESIS_PROMPT | llm

    # Format raw data as readable strings for the LLM
    weather_str = json.dumps(raw_data["weather"], indent=2)
    news_str = json.dumps(raw_data["news"], indent=2)
    finance_str = json.dumps(raw_data["finance"], indent=2)

    result = chain.invoke({
        "weather_data": weather_str,
        "news_data": news_str,
        "finance_data": finance_str,
        "current_date": datetime.now().strftime("%A, %B %d, %Y"),
    })

    return result.content


async def generate_briefing(city: str = "New York") -> tuple[str, dict]:
    """
    Complete pipeline: gather data → synthesize → return report.

    Args:
        city: City for weather data

    Returns:
        Tuple of (formatted report string, raw data dict)
    """
    # Step 1: Gather data in parallel
    raw_data = await gather_all_data(city)

    # Step 2: Synthesize with LLM
    report = synthesize_report(raw_data)

    return report, raw_data
