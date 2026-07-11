"""
Report Generator — Formats raw data and AI synthesis into HTML + Markdown
==========================================================================
Produces both a rich HTML report (for browser viewing) and a clean
Markdown report (for terminal / GitHub viewing).
"""

import json
from datetime import datetime


def generate_html_report(
    report_text: str,
    raw_data: dict,
    city: str = "New York",
) -> str:
    """
    Generate a rich HTML report from the AI-synthesized briefing.

    Args:
        report_text: Markdown-formatted briefing text from the LLM
        raw_data: Raw API data dictionary
        city: City name for the header

    Returns:
        Complete HTML document as a string
    """
    weather = raw_data.get("weather", {})
    news = raw_data.get("news", {})
    finance = raw_data.get("finance", {})

    # Build weather section
    weather_html = ""
    if weather.get("status") == "success":
        weather_html = f"""
        <div class="data-card weather-card">
            <h3>🌤️ Weather in {weather.get('city', city)}</h3>
            <div class="weather-grid">
                <div class="weather-main">
                    <span class="temp">{weather.get('temperature_f', 'N/A')}°F</span>
                    <span class="condition">{weather.get('condition', 'Unknown')}</span>
                </div>
                <div class="weather-details">
                    <p>Feels like: {weather.get('feels_like_f', 'N/A')}°F</p>
                    <p>Humidity: {weather.get('humidity', 'N/A')}%</p>
                    <p>Wind: {weather.get('wind_speed_mph', 'N/A')} mph {weather.get('wind_direction', '')}</p>
                    <p>UV Index: {weather.get('uv_index', 'N/A')}</p>
                </div>
            </div>
        </div>
        """
    else:
        weather_html = '<div class="data-card error-card"><p>⚠️ Weather data unavailable</p></div>'

    # Build news section
    news_html = ""
    stories = news.get("stories", [])
    if stories:
        stories_html = ""
        for story in stories[:8]:
            category_class = story.get("category", "general").lower().replace(" & ", "-").replace(" ", "-")
            stories_html += f"""
            <div class="news-item">
                <span class="category-badge {category_class}">{story.get('category', 'General')}</span>
                <a href="{story.get('url', '#')}" class="story-title">{story.get('title', 'Untitled')}</a>
                <div class="story-meta">
                    <span>⬆ {story.get('score', 0)}</span>
                    <span>💬 {story.get('comments', 0)}</span>
                    <span>by {story.get('author', 'unknown')}</span>
                </div>
            </div>
            """
        news_html = f"""
        <div class="data-card news-card">
            <h3>📰 Top Tech Headlines</h3>
            <div class="news-list">{stories_html}</div>
            <p class="source">Source: {news.get('source', 'Hacker News')}</p>
        </div>
        """
    else:
        news_html = '<div class="data-card error-card"><p>⚠️ News data unavailable</p></div>'

    # Build finance section
    finance_html = ""
    coins = finance.get("coins", [])
    if coins:
        coins_html = ""
        for coin in coins:
            change = coin.get("price_change_24h", 0) or 0
            change_class = "positive" if change >= 0 else "negative"
            change_icon = "📈" if change >= 0 else "📉"
            price = coin.get("current_price", 0) or 0
            coins_html += f"""
            <div class="coin-row">
                <div class="coin-name">
                    <strong>{coin.get('symbol', '???')}</strong>
                    <span class="coin-fullname">{coin.get('name', 'Unknown')}</span>
                </div>
                <div class="coin-price">${price:,.2f}</div>
                <div class="coin-change {change_class}">
                    {change_icon} {change:+.2f}%
                </div>
            </div>
            """

        summary = finance.get("summary", {})
        finance_html = f"""
        <div class="data-card finance-card">
            <h3>💰 Crypto Market Overview</h3>
            <div class="market-sentiment">
                Market Sentiment: <strong>{summary.get('market_sentiment', 'N/A')}</strong>
            </div>
            <div class="coins-list">{coins_html}</div>
            <p class="source">Source: {finance.get('source', 'CoinGecko')}</p>
        </div>
        """
    else:
        finance_html = '<div class="data-card error-card"><p>⚠️ Finance data unavailable</p></div>'

    # Convert markdown-like report text to simple HTML
    report_html = report_text.replace("\n", "<br>")
    for i in range(4, 0, -1):
        marker = "#" * i + " "
        report_html = report_html.replace(
            f"<br>{marker}", f"<br><h{i}>"
        ).replace(marker, f"<h{i}>")
    report_html = report_html.replace("**", "<strong>").replace("- ", "• ")

    now = datetime.now()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Morning Briefing — {now.strftime('%B %d, %Y')}</title>
    <style>
        :root {{
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0b0;
            --accent-blue: #4fc3f7;
            --accent-green: #66bb6a;
            --accent-red: #ef5350;
            --accent-orange: #ffa726;
            --accent-purple: #ab47bc;
            --border-color: #2a2a4a;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            padding: 2rem 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 2.2rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}

        .data-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .data-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .data-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}

        .data-card h3 {{
            color: var(--accent-blue);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}

        .news-card {{
            grid-column: 1 / -1;
        }}

        .weather-grid {{
            display: flex;
            align-items: center;
            gap: 2rem;
        }}

        .weather-main {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .temp {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent-orange);
        }}

        .condition {{
            color: var(--text-secondary);
        }}

        .weather-details p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin: 0.3rem 0;
        }}

        .news-item {{
            padding: 0.8rem 0;
            border-bottom: 1px solid var(--border-color);
        }}

        .news-item:last-child {{
            border-bottom: none;
        }}

        .story-title {{
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 500;
            display: block;
            margin: 0.3rem 0;
        }}

        .story-title:hover {{
            color: var(--accent-blue);
        }}

        .story-meta {{
            display: flex;
            gap: 1rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        .category-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: var(--bg-secondary);
            color: var(--accent-blue);
        }}

        .coin-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--border-color);
        }}

        .coin-row:last-child {{ border-bottom: none; }}

        .coin-name {{ flex: 1; }}
        .coin-fullname {{ color: var(--text-secondary); font-size: 0.85rem; margin-left: 0.5rem; }}
        .coin-price {{ font-weight: bold; margin: 0 1rem; }}
        .coin-change.positive {{ color: var(--accent-green); }}
        .coin-change.negative {{ color: var(--accent-red); }}

        .market-sentiment {{
            text-align: center;
            padding: 0.5rem;
            margin-bottom: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}

        .briefing-section {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 2rem;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }}

        .briefing-section h2 {{
            color: var(--accent-purple);
            margin-bottom: 1rem;
        }}

        .source {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 1rem;
            font-style: italic;
        }}

        .error-card {{
            border-color: var(--accent-red);
            text-align: center;
            color: var(--accent-orange);
        }}

        footer {{
            text-align: center;
            padding: 2rem 0;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
        }}

        @media (max-width: 768px) {{
            .data-grid {{ grid-template-columns: 1fr; }}
            .weather-grid {{ flex-direction: column; }}
            .container {{ padding: 1rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>☀️ Morning Briefing</h1>
            <p class="subtitle">{now.strftime('%A, %B %d, %Y')} · Generated by CalderR API Aggregator Agent</p>
        </header>

        <div class="data-grid">
            {weather_html}
            {finance_html}
            {news_html}
        </div>

        <div class="briefing-section">
            <h2>🤖 AI-Synthesized Briefing</h2>
            <div>{report_html}</div>
        </div>

        <footer>
            <p>Generated at {now.strftime('%I:%M %p')} · CalderR Internship Week 2 · API Aggregator Agent</p>
            <p>Data sources: wttr.in · Hacker News · CoinGecko</p>
        </footer>
    </div>
</body>
</html>"""

    return html


def generate_markdown_report(
    report_text: str,
    raw_data: dict,
    city: str = "New York",
) -> str:
    """
    Generate a clean Markdown report.

    Args:
        report_text: AI-synthesized briefing text
        raw_data: Raw API data dictionary
        city: City name

    Returns:
        Markdown-formatted report string
    """
    now = datetime.now()
    weather = raw_data.get("weather", {})
    news = raw_data.get("news", {})
    finance = raw_data.get("finance", {})

    md = f"""# ☀️ Morning Briefing — {now.strftime('%A, %B %d, %Y')}

> Generated at {now.strftime('%I:%M %p')} by CalderR API Aggregator Agent

---

## 🌤️ Weather in {weather.get('city', city)}

| Metric | Value |
|--------|-------|
| Temperature | {weather.get('temperature_f', 'N/A')}°F ({weather.get('temperature_c', 'N/A')}°C) |
| Feels Like | {weather.get('feels_like_f', 'N/A')}°F |
| Condition | {weather.get('condition', 'Unknown')} |
| Humidity | {weather.get('humidity', 'N/A')}% |
| Wind | {weather.get('wind_speed_mph', 'N/A')} mph {weather.get('wind_direction', '')} |
| UV Index | {weather.get('uv_index', 'N/A')} |

---

## 📰 Top Tech Headlines

"""
    for i, story in enumerate(news.get("stories", [])[:8], 1):
        score = story.get("score", 0)
        comments = story.get("comments", 0)
        md += f"{i}. **{story.get('title', 'Untitled')}** [{story.get('category', 'General')}]\n"
        md += f"   ⬆ {score} · 💬 {comments} · by {story.get('author', 'unknown')}\n\n"

    md += "---\n\n## 💰 Crypto Market\n\n"
    md += "| Coin | Price (USD) | 24h Change |\n"
    md += "|------|-------------|------------|\n"

    for coin in finance.get("coins", []):
        price = coin.get("current_price", 0) or 0
        change = coin.get("price_change_24h", 0) or 0
        icon = "📈" if change >= 0 else "📉"
        md += f"| {coin.get('symbol', '???')} | ${price:,.2f} | {icon} {change:+.2f}% |\n"

    summary = finance.get("summary", {})
    md += f"\nMarket Sentiment: **{summary.get('market_sentiment', 'N/A')}**\n"

    md += f"""
---

## 🤖 AI-Synthesized Briefing

{report_text}

---

*Data sources: wttr.in · Hacker News · CoinGecko*
*Generated by CalderR Internship Week 2 — API Aggregator Agent*
"""

    return md
