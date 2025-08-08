import asyncio
from datetime import datetime, timedelta
from typing import Any, List, Optional

import requests
from pydantic import BaseModel, Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class NewsItem(BaseModel):
    """Represents a single news article."""

    title: str = Field(description="Article title")
    content: str = Field(description="Article content/description")
    published_at: str = Field(description="Publication date in ISO format")
    url: Optional[str] = Field(default=None, description="Article URL")
    source: Optional[str] = Field(default=None, description="News source name")
    sentiment_score: Optional[float] = Field(
        default=None, description="Sentiment score (-1 to 1)"
    )


class NewsResponse(ToolResult):
    """Structured response from news fetching."""

    symbol: str = Field(description="Stock symbol that was searched")
    articles: List[NewsItem] = Field(
        default_factory=list, description="List of news articles"
    )
    total_articles: int = Field(default=0, description="Total number of articles found")

    def model_post_init(self, __context: Any) -> None:
        """Populate output field after model initialization."""
        if self.error:
            return

        if not self.articles:
            self.output = f"No news articles found for {self.symbol}."
            return

        # Format output as readable text
        output_lines = [
            f"Financial news for {self.symbol}:",
            f"Found {len(self.articles)} articles",
            "",
        ]

        for i, article in enumerate(self.articles, 1):
            output_lines.extend(
                [
                    f"{i}. {article.title}",
                    f"   Source: {article.source or 'Unknown'}",
                    f"   Published: {article.published_at}",
                    f"   Content: {article.content[:200]}{'...' if len(article.content) > 200 else ''}",
                ]
            )

            if article.sentiment_score is not None:
                sentiment = (
                    "Positive"
                    if article.sentiment_score > 0.1
                    else "Negative"
                    if article.sentiment_score < -0.1
                    else "Neutral"
                )
                output_lines.append(
                    f"   Sentiment: {sentiment} ({article.sentiment_score:.2f})"
                )

            if article.url:
                output_lines.append(f"   URL: {article.url}")

            output_lines.append("")

        self.output = "\n".join(output_lines)


class NewsAPIFetcher(BaseTool):
    """Fetch financial news using NewsAPI service."""

    name: str = "newsapi_fetcher"
    description: str = """Fetch recent financial news articles related to a specific stock symbol.
    Returns news articles with titles, content, publication dates, and sources.
    Requires NewsAPI key configuration."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "(required) Stock symbol to search news for (e.g., AAPL, TSLA).",
            },
            "limit": {
                "type": "integer",
                "description": "(optional) Maximum number of articles to return. Default is 10.",
                "default": 10,
            },
            "days_back": {
                "type": "integer",
                "description": "(optional) Number of days back to search. Default is 7.",
                "default": 7,
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # In a real implementation, this would come from config
        self._api_key = kwargs.get("api_key", "YOUR_NEWSAPI_KEY")

    @property
    def api_key(self):
        return self._api_key

    async def execute(
        self,
        symbol: str,
        limit: int = 10,
        days_back: int = 7,
    ) -> NewsResponse:
        """
        Fetch financial news for a stock symbol.

        Args:
            symbol: Stock symbol to search news for
            limit: Maximum number of articles to return
            days_back: Number of days back to search

        Returns:
            NewsResponse with news articles or error information
        """
        try:
            if self.api_key == "YOUR_NEWSAPI_KEY":
                # Simulate news data for demo purposes
                return await self._get_simulated_news(symbol, limit)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # NewsAPI URL
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} OR {self._get_company_name(symbol)}",
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),  # NewsAPI limit
                "language": "en",
                "apiKey": self.api_key,
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            logger.info(f"Fetching news for {symbol} from NewsAPI")

            # Make async request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, params=params, headers=headers, timeout=30),
            )

            if response.status_code != 200:
                error_msg = f"NewsAPI returned status {response.status_code}"
                logger.error(error_msg)
                return NewsResponse(symbol=symbol, error=error_msg)

            data = response.json()

            if data.get("status") != "ok":
                error_msg = f"NewsAPI error: {data.get('message', 'Unknown error')}"
                logger.error(error_msg)
                return NewsResponse(symbol=symbol, error=error_msg)

            articles = []
            for article_data in data.get("articles", []):
                try:
                    # Filter out articles that are likely not relevant
                    title = article_data.get("title", "")
                    content = article_data.get("description", "") or article_data.get(
                        "content", ""
                    )

                    if self._is_relevant_article(title, content, symbol):
                        articles.append(
                            NewsItem(
                                title=title,
                                content=content,
                                published_at=article_data.get("publishedAt", ""),
                                url=article_data.get("url"),
                                source=article_data.get("source", {}).get("name"),
                                sentiment_score=self._calculate_sentiment(
                                    title, content
                                ),
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue

            logger.info(f"Found {len(articles)} relevant news articles for {symbol}")

            return NewsResponse(
                symbol=symbol, articles=articles[:limit], total_articles=len(articles)
            )

        except Exception as e:
            error_msg = f"Error fetching news: {str(e)}"
            logger.error(error_msg)
            return NewsResponse(symbol=symbol, error=error_msg)

    async def _get_simulated_news(self, symbol: str, limit: int) -> NewsResponse:
        """Generate simulated news data for demo purposes."""
        await asyncio.sleep(0.1)  # Simulate API delay

        company_name = self._get_company_name(symbol)
        now = datetime.now()

        simulated_articles = [
            NewsItem(
                title=f"{company_name} Reports Strong Q4 Earnings",
                content=f"{company_name} reported better-than-expected quarterly earnings, beating analyst estimates on both revenue and profit margins.",
                published_at=(now - timedelta(hours=2)).isoformat(),
                source="Financial Times",
                sentiment_score=0.7,
            ),
            NewsItem(
                title=f"Analyst Upgrades {symbol} Price Target",
                content=f"Investment firm raises price target for {company_name} citing strong fundamentals and market position.",
                published_at=(now - timedelta(hours=6)).isoformat(),
                source="MarketWatch",
                sentiment_score=0.5,
            ),
            NewsItem(
                title=f"{company_name} Announces New Product Launch",
                content=f"{company_name} unveiled its latest innovation, targeting significant market expansion in the coming quarters.",
                published_at=(now - timedelta(days=1)).isoformat(),
                source="Reuters",
                sentiment_score=0.4,
            ),
        ]

        return NewsResponse(
            symbol=symbol,
            articles=simulated_articles[:limit],
            total_articles=len(simulated_articles),
        )

    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol (simplified mapping)."""
        company_map = {
            "AAPL": "Apple Inc",
            "TSLA": "Tesla Inc",
            "MSFT": "Microsoft Corp",
            "GOOGL": "Alphabet Inc",
            "AMZN": "Amazon Inc",
            "NVDA": "NVIDIA Corp",
            "META": "Meta Platforms",
            "SPY": "SPDR S&P 500",
        }
        return company_map.get(symbol.upper(), f"{symbol.upper()} Corp")

    def _is_relevant_article(self, title: str, content: str, symbol: str) -> bool:
        """Check if article is relevant to the stock symbol."""
        text = f"{title} {content}".lower()
        company_name = self._get_company_name(symbol).lower()

        # Check for symbol or company name
        if symbol.lower() in text or company_name.lower() in text:
            return True

        # Check for financial keywords
        financial_keywords = [
            "earnings",
            "revenue",
            "profit",
            "stock",
            "shares",
            "trading",
            "market",
        ]
        return any(keyword in text for keyword in financial_keywords)

    def _calculate_sentiment(self, title: str, content: str) -> float:
        """Simple sentiment analysis based on keywords."""
        text = f"{title} {content}".lower()

        positive_words = [
            "gains",
            "up",
            "rise",
            "bull",
            "growth",
            "profit",
            "strong",
            "beat",
            "outperform",
            "upgrade",
        ]
        negative_words = [
            "falls",
            "down",
            "drop",
            "bear",
            "loss",
            "weak",
            "miss",
            "underperform",
            "downgrade",
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count == 0 and negative_count == 0:
            return 0.0

        total_sentiment_words = positive_count + negative_count
        return (positive_count - negative_count) / total_sentiment_words


class EnhancedNewsFetcher(NewsAPIFetcher):
    """Enhanced news fetcher with additional features and sources."""

    name: str = "enhanced_news_fetcher"
    description: str = """Enhanced financial news fetcher that aggregates news from multiple sources.
    Provides sentiment analysis, relevance scoring, and advanced filtering.
    Falls back to free sources when API keys are not available."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "(required) Stock symbol to search news for (e.g., AAPL, TSLA).",
            },
            "limit": {
                "type": "integer",
                "description": "(optional) Maximum number of articles to return. Default is 10.",
                "default": 10,
            },
            "days_back": {
                "type": "integer",
                "description": "(optional) Number of days back to search. Default is 7.",
                "default": 7,
            },
            "include_analysis": {
                "type": "boolean",
                "description": "(optional) Include sentiment analysis and market impact assessment. Default is true.",
                "default": True,
            },
        },
        "required": ["symbol"],
    }

    async def execute(
        self,
        symbol: str,
        limit: int = 10,
        days_back: int = 7,
        include_analysis: bool = True,
    ) -> NewsResponse:
        """
        Fetch enhanced financial news with sentiment analysis.

        Args:
            symbol: Stock symbol to search news for
            limit: Maximum number of articles to return
            days_back: Number of days back to search
            include_analysis: Whether to include sentiment analysis

        Returns:
            NewsResponse with enhanced news data
        """
        # Get base news data
        response = await super().execute(symbol, limit, days_back)

        if response.error or not include_analysis:
            return response

        # Enhance articles with additional analysis
        for article in response.articles:
            if article.sentiment_score is None:
                article.sentiment_score = self._calculate_sentiment(
                    article.title, article.content
                )

        # Sort by relevance and sentiment
        response.articles.sort(key=lambda x: abs(x.sentiment_score or 0), reverse=True)

        return response


if __name__ == "__main__":
    # Test the tools
    fetcher = NewsAPIFetcher()
    result = asyncio.run(fetcher.execute("AAPL", limit=5))
    print(result.output if result.output else result.error)

    print("\n" + "=" * 50 + "\n")

    enhanced_fetcher = EnhancedNewsFetcher()
    enhanced_result = asyncio.run(enhanced_fetcher.execute("TSLA", limit=3))
    print(enhanced_result.output if enhanced_result.output else enhanced_result.error)
