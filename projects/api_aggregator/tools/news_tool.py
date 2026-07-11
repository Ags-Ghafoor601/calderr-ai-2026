"""
News Tool — Fetches real-time news from Hacker News API
=========================================================
Hacker News (news.ycombinator.com) provides a free, public, no-API-key
REST API for accessing top stories, new stories, and story details.
"""

import httpx
import asyncio
from datetime import datetime


HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"


async def _fetch_story(client: httpx.AsyncClient, story_id: int) -> dict:
    """Fetch a single story's details by ID."""
    try:
        response = await client.get(
            f"{HN_BASE_URL}/item/{story_id}.json",
            timeout=8.0,
        )
        response.raise_for_status()
        data = response.json()
        if data is None:
            return {}

        return {
            "title": data.get("title", "Untitled"),
            "url": data.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            "score": data.get("score", 0),
            "author": data.get("by", "unknown"),
            "comments": data.get("descendants", 0),
            "time": datetime.fromtimestamp(
                data.get("time", 0)
            ).strftime("%Y-%m-%d %H:%M") if data.get("time") else "Unknown",
            "hn_link": f"https://news.ycombinator.com/item?id={story_id}",
            "type": data.get("type", "story"),
        }
    except Exception:
        return {}


async def fetch_news(num_stories: int = 10) -> dict:
    """
    Fetch top news stories from Hacker News (no API key needed).

    Args:
        num_stories: Number of top stories to fetch (max 30)

    Returns:
        Dictionary with list of stories and metadata.
    """
    num_stories = min(num_stories, 30)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch top story IDs
            response = await client.get(f"{HN_BASE_URL}/topstories.json")
            response.raise_for_status()
            story_ids = response.json()[:num_stories]

            # Fetch all stories in parallel
            tasks = [_fetch_story(client, sid) for sid in story_ids]
            stories = await asyncio.gather(*tasks)

        # Filter out empty results and sort by score
        valid_stories = [s for s in stories if s]
        valid_stories.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Categorize stories by inferring topics
        categorized = []
        for story in valid_stories:
            title_lower = story["title"].lower()
            category = "General"
            if any(w in title_lower for w in ["ai", "llm", "gpt", "ml", "neural", "model"]):
                category = "AI & Machine Learning"
            elif any(w in title_lower for w in ["python", "rust", "go", "java", "code", "programming", "developer"]):
                category = "Programming"
            elif any(w in title_lower for w in ["startup", "funding", "vc", "billion", "acquisition"]):
                category = "Business & Startups"
            elif any(w in title_lower for w in ["linux", "kernel", "os", "hardware", "chip"]):
                category = "Systems & Hardware"
            elif any(w in title_lower for w in ["security", "hack", "vulnerability", "breach", "privacy"]):
                category = "Security"
            elif any(w in title_lower for w in ["web", "browser", "css", "html", "react", "frontend"]):
                category = "Web Development"

            story["category"] = category
            categorized.append(story)

        return {
            "stories": categorized,
            "total_fetched": len(categorized),
            "source": "Hacker News (news.ycombinator.com)",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
        }

    except httpx.TimeoutException:
        return {
            "stories": [],
            "status": "error",
            "error": "Request timed out — Hacker News API unreachable",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {
            "stories": [],
            "status": "error",
            "error": str(e),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
