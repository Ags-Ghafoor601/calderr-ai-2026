"""
Finance Tool — Fetches cryptocurrency data from CoinGecko API
================================================================
CoinGecko provides a free, no-API-key public API for cryptocurrency
market data including prices, market cap, and 24h changes.
"""

import httpx
from datetime import datetime


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Top cryptocurrencies to track
DEFAULT_COINS = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"]


async def fetch_finance(
    coins: list[str] | None = None,
    vs_currency: str = "usd",
) -> dict:
    """
    Fetch cryptocurrency market data from CoinGecko (no API key needed).

    Args:
        coins: List of coin IDs (e.g., ["bitcoin", "ethereum"]).
               Defaults to top 5 cryptocurrencies.
        vs_currency: Fiat currency for prices (default: "usd")

    Returns:
        Dictionary with market data for each coin.
    """
    if coins is None:
        coins = DEFAULT_COINS

    coin_ids = ",".join(coins)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch market data
            response = await client.get(
                f"{COINGECKO_BASE_URL}/coins/markets",
                params={
                    "vs_currency": vs_currency,
                    "ids": coin_ids,
                    "order": "market_cap_desc",
                    "sparkline": "false",
                    "price_change_percentage": "24h,7d",
                },
            )
            response.raise_for_status()
            data = response.json()

        market_data = []
        for coin in data:
            market_data.append({
                "name": coin.get("name", "Unknown"),
                "symbol": coin.get("symbol", "???").upper(),
                "current_price": coin.get("current_price", 0),
                "market_cap": coin.get("market_cap", 0),
                "market_cap_rank": coin.get("market_cap_rank", "N/A"),
                "total_volume_24h": coin.get("total_volume", 0),
                "price_change_24h": coin.get("price_change_percentage_24h", 0),
                "price_change_7d": coin.get(
                    "price_change_percentage_7d_in_currency", 0
                ),
                "high_24h": coin.get("high_24h", 0),
                "low_24h": coin.get("low_24h", 0),
                "ath": coin.get("ath", 0),
                "ath_change_pct": coin.get("ath_change_percentage", 0),
                "last_updated": coin.get("last_updated", "Unknown"),
            })

        # Calculate portfolio summary
        total_market_cap = sum(c.get("market_cap", 0) for c in market_data)
        avg_change_24h = (
            sum(c.get("price_change_24h", 0) for c in market_data) / len(market_data)
            if market_data else 0
        )

        return {
            "coins": market_data,
            "summary": {
                "total_coins_tracked": len(market_data),
                "total_market_cap": total_market_cap,
                "avg_24h_change": round(avg_change_24h, 2),
                "market_sentiment": (
                    "Bullish 📈" if avg_change_24h > 1
                    else "Bearish 📉" if avg_change_24h < -1
                    else "Neutral ➡️"
                ),
            },
            "currency": vs_currency.upper(),
            "source": "CoinGecko",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
        }

    except httpx.TimeoutException:
        return {
            "coins": [],
            "status": "error",
            "error": "Request timed out — CoinGecko API unreachable",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except httpx.HTTPStatusError as e:
        return {
            "coins": [],
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {
            "coins": [],
            "status": "error",
            "error": str(e),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
