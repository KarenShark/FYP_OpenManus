import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class MarketData(BaseModel):
    """Represents market data for a single trading day."""

    date: str = Field(description="Trading date in YYYY-MM-DD format")
    open: float = Field(description="Opening price")
    high: float = Field(description="Highest price")
    low: float = Field(description="Lowest price")
    close: float = Field(description="Closing price")
    volume: int = Field(description="Trading volume")


class YahooFinanceResponse(ToolResult):
    """Structured response from Yahoo Finance data fetch."""

    symbol: str = Field(description="Stock symbol that was fetched")
    data: List[MarketData] = Field(
        default_factory=list, description="List of market data points"
    )
    start_date: str = Field(description="Start date of the data range")
    end_date: str = Field(description="End date of the data range")

    def model_post_init(self, __context: Any) -> None:
        """Populate output field after model initialization."""
        if self.error:
            return

        if not self.data:
            self.output = (
                f"No market data found for {self.symbol} in the specified date range."
            )
            return

        # Format output as readable text
        output_lines = [
            f"Market data for {self.symbol} ({self.start_date} to {self.end_date}):",
            f"Total trading days: {len(self.data)}",
            "",
        ]

        # Add data summary
        if self.data:
            latest = self.data[-1]
            earliest = self.data[0]

            output_lines.extend(
                [
                    f"Latest trading day ({latest.date}):",
                    f"  Open: ${latest.open:.2f}",
                    f"  High: ${latest.high:.2f}",
                    f"  Low: ${latest.low:.2f}",
                    f"  Close: ${latest.close:.2f}",
                    f"  Volume: {latest.volume:,}",
                    "",
                    f"Period performance:",
                    f"  Start price: ${earliest.open:.2f}",
                    f"  End price: ${latest.close:.2f}",
                    f"  Change: ${latest.close - earliest.open:.2f} ({((latest.close - earliest.open) / earliest.open * 100):+.2f}%)",
                ]
            )

        self.output = "\n".join(output_lines)


class YahooFinanceFetcher(BaseTool):
    """Fetch historical market data from Yahoo Finance."""

    name: str = "yfinance_fetcher"
    description: str = """Fetch historical stock market data from Yahoo Finance.
    Returns daily OHLCV (Open, High, Low, Close, Volume) data for a specified symbol and date range.
    Supports major stock symbols, ETFs, and indices."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "(required) Stock symbol to fetch data for (e.g., AAPL, TSLA, SPY).",
            },
            "start_date": {
                "type": "string",
                "description": "(optional) Start date in YYYY-MM-DD format. Defaults to 30 days ago.",
            },
            "end_date": {
                "type": "string",
                "description": "(optional) End date in YYYY-MM-DD format. Defaults to today.",
            },
        },
        "required": ["symbol"],
    }

    async def execute(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> YahooFinanceResponse:
        """
        Fetch historical market data from Yahoo Finance.

        Args:
            symbol: Stock symbol to fetch data for
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            YahooFinanceResponse with market data or error information
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

            # Yahoo Finance API URL
            url = (
                f"https://query1.finance.yahoo.com/v7/finance/download/{symbol.upper()}"
            )
            params = {
                "period1": start_timestamp,
                "period2": end_timestamp,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            logger.info(
                f"Fetching Yahoo Finance data for {symbol} from {start_date} to {end_date}"
            )

            # Make async request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, params=params, headers=headers, timeout=30),
            )

            if response.status_code != 200:
                error_msg = f"Yahoo Finance API returned status {response.status_code}"
                logger.error(error_msg)
                return YahooFinanceResponse(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    error=error_msg,
                )

            # Parse CSV response
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                error_msg = "No data returned from Yahoo Finance"
                logger.warning(error_msg)
                return YahooFinanceResponse(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    error=error_msg,
                )

            # Parse header and data rows
            header = lines[0].split(",")
            data_rows = lines[1:]

            market_data = []
            for row in data_rows:
                try:
                    values = row.split(",")
                    if (
                        len(values) >= 6 and values[1] != "null"
                    ):  # Skip rows with missing data
                        market_data.append(
                            MarketData(
                                date=values[0],
                                open=float(values[1]),
                                high=float(values[2]),
                                low=float(values[3]),
                                close=float(values[4]),
                                volume=int(
                                    float(values[6])
                                ),  # Convert to int, handling scientific notation
                            )
                        )
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid data row: {row} - {e}")
                    continue

            if not market_data:
                error_msg = f"No valid market data found for {symbol}"
                logger.warning(error_msg)
                return YahooFinanceResponse(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    error=error_msg,
                )

            # Sort by date to ensure chronological order
            market_data.sort(key=lambda x: x.date)

            logger.info(
                f"Successfully fetched {len(market_data)} data points for {symbol}"
            )

            return YahooFinanceResponse(
                symbol=symbol,
                data=market_data,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            error_msg = f"Error fetching Yahoo Finance data: {str(e)}"
            logger.error(error_msg)
            return YahooFinanceResponse(
                symbol=symbol,
                start_date=start_date or "",
                end_date=end_date or "",
                error=error_msg,
            )


if __name__ == "__main__":
    # Test the tool
    fetcher = YahooFinanceFetcher()
    result = asyncio.run(fetcher.execute("AAPL", "2024-01-01", "2024-01-31"))
    print(result.output if result.output else result.error)
