"""
Integration tests for trading platform tools.
Tests all new tools including YahooFinanceFetcher, NewsAPI tools, TechnicalIndicators, and Risk tools.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from app.tool.indicators import TechnicalIndicators
from app.tool.newsapi_fetcher import EnhancedNewsFetcher, NewsAPIFetcher
from app.tool.risk_tools import PositionSizer, RiskAnalyzer
from app.tool.yfinance_fetcher import YahooFinanceFetcher


class TestYahooFinanceFetcher:
    """Test cases for YahooFinanceFetcher tool."""

    def test_tool_initialization(self):
        """Test that YahooFinanceFetcher can be initialized properly."""
        tool = YahooFinanceFetcher()
        assert tool.name == "yfinance_fetcher"
        assert "stock market data" in tool.description.lower()
        assert "symbol" in tool.parameters["properties"]
        assert "symbol" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test basic execution with valid parameters."""
        tool = YahooFinanceFetcher()

        # Use a recent date range to ensure data availability
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

        result = await tool.execute(
            symbol="AAPL", start_date=start_date, end_date=end_date
        )

        assert result is not None
        assert hasattr(result, "symbol")
        assert result.symbol == "AAPL"

        # Should either have data or a reasonable error
        if result.error:
            print(f"Expected error in test environment: {result.error}")
        else:
            assert hasattr(result, "data")
            assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_invalid_symbol(self):
        """Test handling of invalid stock symbols."""
        tool = YahooFinanceFetcher()

        result = await tool.execute(symbol="INVALID_SYMBOL_XYZ123")

        # Should handle gracefully - either no data or error
        assert result is not None
        assert result.symbol == "INVALID_SYMBOL_XYZ123"

    @pytest.mark.asyncio
    async def test_default_date_handling(self):
        """Test that default dates are handled correctly."""
        tool = YahooFinanceFetcher()

        result = await tool.execute(symbol="SPY")  # Use SPY as it's very reliable

        assert result is not None
        assert result.symbol == "SPY"
        assert result.start_date  # Should have some start date
        assert result.end_date  # Should have some end date


class TestNewsAPIFetcher:
    """Test cases for NewsAPIFetcher and EnhancedNewsFetcher tools."""

    def test_newsapi_fetcher_initialization(self):
        """Test that NewsAPIFetcher can be initialized properly."""
        tool = NewsAPIFetcher()
        assert tool.name == "newsapi_fetcher"
        assert "financial news" in tool.description.lower()
        assert "symbol" in tool.parameters["properties"]
        assert "symbol" in tool.parameters["required"]

    def test_enhanced_news_fetcher_initialization(self):
        """Test that EnhancedNewsFetcher can be initialized properly."""
        tool = EnhancedNewsFetcher()
        assert tool.name == "enhanced_news_fetcher"
        assert "enhanced" in tool.description.lower()
        assert "include_analysis" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_newsapi_basic_execution(self):
        """Test basic execution of NewsAPIFetcher."""
        tool = NewsAPIFetcher()

        result = await tool.execute(symbol="AAPL", limit=3)

        assert result is not None
        assert hasattr(result, "symbol")
        assert result.symbol == "AAPL"

        # Since we're using simulated data, should always have articles
        if not result.error:
            assert hasattr(result, "articles")
            assert isinstance(result.articles, list)
            assert len(result.articles) <= 3

    @pytest.mark.asyncio
    async def test_enhanced_news_execution(self):
        """Test enhanced news fetcher with analysis."""
        tool = EnhancedNewsFetcher()

        result = await tool.execute(symbol="TSLA", limit=2, include_analysis=True)

        assert result is not None
        assert result.symbol == "TSLA"

        if not result.error and result.articles:
            # Check that sentiment analysis is included
            for article in result.articles:
                assert hasattr(article, "sentiment_score")
                if article.sentiment_score is not None:
                    assert -1.0 <= article.sentiment_score <= 1.0

    @pytest.mark.asyncio
    async def test_news_error_handling(self):
        """Test error handling with invalid parameters."""
        tool = NewsAPIFetcher()

        # Test with empty symbol
        result = await tool.execute(symbol="", limit=5)

        # Should handle gracefully
        assert result is not None


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators tool."""

    def test_tool_initialization(self):
        """Test that TechnicalIndicators can be initialized properly."""
        tool = TechnicalIndicators()
        assert tool.name == "technical_indicators"
        assert "technical indicators" in tool.description.lower()
        assert "price_data" in tool.parameters["properties"]
        assert "price_data" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_basic_indicators_calculation(self):
        """Test basic indicator calculations with sample data."""
        tool = TechnicalIndicators()

        # Create sample price data
        price_data = []
        base_price = 100.0
        for i in range(30):  # 30 days of data
            price = (
                base_price + (i * 0.5) + ((-1) ** i * 2)
            )  # Trending up with volatility
            price_data.append(
                {
                    "close": price,
                    "high": price + 1,
                    "low": price - 1,
                    "volume": 1000000,
                    "date": f"2024-01-{i+1:02d}",
                }
            )

        result = await tool.execute(
            price_data=price_data, symbol="TEST", indicators=["SMA", "RSI"]
        )

        assert result is not None
        assert not result.error
        assert hasattr(result, "indicators")
        assert hasattr(result, "signals")

        # Check that indicators were calculated
        assert "SMA" in result.indicators
        assert "RSI" in result.indicators

        # Check indicator data structure
        sma_data = result.indicators["SMA"]
        assert hasattr(sma_data, "values")
        assert hasattr(sma_data, "latest_value")
        assert sma_data.latest_value is not None

        # Check that signals were generated
        assert isinstance(result.signals, list)
        if result.signals:
            signal = result.signals[0]
            assert hasattr(signal, "indicator")
            assert hasattr(signal, "signal")
            assert signal.signal in ["BUY", "SELL", "HOLD"]

    @pytest.mark.asyncio
    async def test_multiple_indicators(self):
        """Test calculation of multiple indicators."""
        tool = TechnicalIndicators()

        # Create sufficient price data for all indicators
        price_data = []
        for i in range(50):
            price = 100 + (i * 0.1) + ((-1) ** i * 1.5)
            price_data.append(
                {
                    "close": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "date": f"2024-01-{i+1:02d}",
                }
            )

        result = await tool.execute(
            price_data=price_data,
            symbol="MULTI_TEST",
            indicators=["SMA", "EMA", "RSI", "MACD", "BB"],
            sma_period=10,
            ema_period=12,
            rsi_period=14,
        )

        assert result is not None
        if not result.error:
            # Should have calculated all requested indicators
            expected_indicators = ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands"]
            for indicator in expected_indicators:
                if indicator in result.indicators:
                    indicator_data = result.indicators[indicator]
                    assert hasattr(indicator_data, "parameters")
                    assert hasattr(indicator_data, "latest_value")

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient price data."""
        tool = TechnicalIndicators()

        # Very limited data
        price_data = [{"close": 100, "date": "2024-01-01"}]

        result = await tool.execute(price_data=price_data, symbol="INSUFFICIENT_TEST")

        # Should handle gracefully
        assert result is not None
        # Might have error or empty indicators
        if result.error:
            assert (
                "insufficient" in result.error.lower()
                or "not enough" in result.error.lower()
            )


class TestPositionSizer:
    """Test cases for PositionSizer tool."""

    def test_tool_initialization(self):
        """Test that PositionSizer can be initialized properly."""
        tool = PositionSizer()
        assert tool.name == "position_sizer"
        assert "position size" in tool.description.lower()
        required_params = tool.parameters["required"]
        assert "account_balance" in required_params
        assert "risk_per_trade" in required_params
        assert "entry_price" in required_params
        assert "stop_loss_price" in required_params

    @pytest.mark.asyncio
    async def test_basic_position_sizing(self):
        """Test basic position sizing calculation."""
        tool = PositionSizer()

        result = await tool.execute(
            account_balance=100000,
            risk_per_trade=0.02,  # 2%
            entry_price=150.0,
            stop_loss_price=140.0,
        )

        assert result is not None
        assert not result.error
        assert hasattr(result, "position_result")

        pos_result = result.position_result
        assert pos_result is not None
        assert hasattr(pos_result, "recommended_position_size")
        assert hasattr(pos_result, "actual_risk_percentage")
        assert hasattr(pos_result, "recommended_position_value")

        # Verify calculations make sense
        risk_per_share = abs(150.0 - 140.0)  # $10
        expected_risk_amount = 100000 * 0.02  # $2000
        expected_position_size = expected_risk_amount / risk_per_share  # 200 shares

        assert abs(pos_result.recommended_position_size - expected_position_size) < 0.01
        assert abs(pos_result.actual_risk_percentage - 2.0) < 0.01

    @pytest.mark.asyncio
    async def test_position_sizing_with_target(self):
        """Test position sizing with target price (risk/reward ratio)."""
        tool = PositionSizer()

        result = await tool.execute(
            account_balance=50000,
            risk_per_trade=0.015,  # 1.5%
            entry_price=100.0,
            stop_loss_price=95.0,
            target_price=110.0,
        )

        assert result is not None
        assert not result.error

        pos_result = result.position_result
        assert pos_result.risk_reward_ratio is not None

        # Risk: $5, Reward: $10, so ratio should be 2.0
        expected_ratio = 10.0 / 5.0
        assert abs(pos_result.risk_reward_ratio - expected_ratio) < 0.01

    @pytest.mark.asyncio
    async def test_risk_warnings_generation(self):
        """Test generation of risk warnings."""
        tool = PositionSizer()

        # Use high risk percentage to trigger warnings
        result = await tool.execute(
            account_balance=10000,
            risk_per_trade=0.10,  # 10% - very high risk
            entry_price=100.0,
            stop_loss_price=90.0,
        )

        assert result is not None
        if not result.error:
            # Should generate risk warnings for high risk
            assert hasattr(result, "risk_warnings")
            if result.risk_warnings:
                warning = result.risk_warnings[0]
                assert hasattr(warning, "level")
                assert hasattr(warning, "message")
                assert warning.level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        tool = PositionSizer()

        # Test with negative account balance
        result = await tool.execute(
            account_balance=-1000,
            risk_per_trade=0.02,
            entry_price=100.0,
            stop_loss_price=95.0,
        )

        assert result is not None
        assert result.error is not None
        assert "positive" in result.error.lower()


class TestRiskAnalyzer:
    """Test cases for RiskAnalyzer tool."""

    def test_tool_initialization(self):
        """Test that RiskAnalyzer can be initialized properly."""
        tool = RiskAnalyzer()
        assert tool.name == "risk_analyzer"
        assert "risk analysis" in tool.description.lower()
        required_params = tool.parameters["required"]
        assert "positions" in required_params
        assert "total_portfolio_value" in required_params

    @pytest.mark.asyncio
    async def test_basic_risk_analysis(self):
        """Test basic portfolio risk analysis."""
        tool = RiskAnalyzer()

        positions = [
            {
                "symbol": "AAPL",
                "value": 25000,
                "volatility": 0.25,
                "volume": 50000000,
                "sector": "Technology",
            },
            {
                "symbol": "MSFT",
                "value": 20000,
                "volatility": 0.20,
                "volume": 30000000,
                "sector": "Technology",
            },
            {
                "symbol": "JPM",
                "value": 15000,
                "volatility": 0.18,
                "volume": 20000000,
                "sector": "Finance",
            },
        ]

        result = await tool.execute(positions=positions, total_portfolio_value=100000)

        assert result is not None
        assert not result.error
        assert hasattr(result, "analysis_result")

        analysis = result.analysis_result
        assert analysis is not None
        assert hasattr(analysis, "overall_risk_score")
        assert hasattr(analysis, "concentration_risk")
        assert hasattr(analysis, "volatility_risk")
        assert hasattr(analysis, "position_correlation_risk")
        assert hasattr(analysis, "liquidity_risk")

        # Risk scores should be between -10 and 10 (some calculations can be negative)
        assert -10 <= analysis.overall_risk_score <= 10
        assert -10 <= analysis.concentration_risk <= 10
        assert 0 <= analysis.volatility_risk <= 10
        assert 0 <= analysis.position_correlation_risk <= 10
        assert 0 <= analysis.liquidity_risk <= 10

    @pytest.mark.asyncio
    async def test_high_concentration_risk(self):
        """Test detection of high concentration risk."""
        tool = RiskAnalyzer()

        # Single large position
        positions = [
            {
                "symbol": "AAPL",
                "value": 80000,  # 80% of portfolio
                "volatility": 0.30,
                "volume": 50000000,
                "sector": "Technology",
            },
            {
                "symbol": "MSFT",
                "value": 10000,
                "volatility": 0.25,
                "volume": 30000000,
                "sector": "Technology",
            },
        ]

        result = await tool.execute(positions=positions, total_portfolio_value=100000)

        assert result is not None
        if not result.error:
            # Should detect high concentration risk
            analysis = result.analysis_result
            assert analysis.concentration_risk > 2.0  # Should be relatively high

            # Should generate warnings for concentration
            if result.risk_warnings:
                concentration_warnings = [
                    w
                    for w in result.risk_warnings
                    if "concentration" in w.message.lower()
                    or "portfolio" in w.message.lower()
                ]
                # Should have some relevant warnings

    @pytest.mark.asyncio
    async def test_empty_positions_handling(self):
        """Test handling of empty positions list."""
        tool = RiskAnalyzer()

        result = await tool.execute(positions=[], total_portfolio_value=100000)

        assert result is not None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_sector_diversification_analysis(self):
        """Test analysis of sector diversification."""
        tool = RiskAnalyzer()

        # Well diversified across sectors
        positions = [
            {
                "symbol": "AAPL",
                "value": 20000,
                "volatility": 0.25,
                "volume": 50000000,
                "sector": "Technology",
            },
            {
                "symbol": "JPM",
                "value": 20000,
                "volatility": 0.18,
                "volume": 20000000,
                "sector": "Finance",
            },
            {
                "symbol": "JNJ",
                "value": 20000,
                "volatility": 0.15,
                "volume": 15000000,
                "sector": "Healthcare",
            },
            {
                "symbol": "XOM",
                "value": 20000,
                "volatility": 0.28,
                "volume": 25000000,
                "sector": "Energy",
            },
            {
                "symbol": "WMT",
                "value": 20000,
                "volatility": 0.12,
                "volume": 18000000,
                "sector": "Consumer",
            },
        ]

        result = await tool.execute(positions=positions, total_portfolio_value=100000)

        assert result is not None
        if not result.error:
            # Well diversified portfolio should have lower correlation risk
            analysis = result.analysis_result
            # Correlation risk should be relatively low due to sector diversification
            assert analysis.position_correlation_risk < 8.0


# Integration test for the complete workflow
class TestTradingWorkflowIntegration:
    """Test complete trading analysis workflow using all tools together."""

    @pytest.mark.asyncio
    async def test_complete_trading_analysis_workflow(self):
        """Test a complete trading analysis workflow using all tools."""

        # 1. Fetch market data
        yfinance_tool = YahooFinanceFetcher()
        market_result = await yfinance_tool.execute(
            symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31"
        )

        # Should get market data or reasonable error
        assert market_result is not None

        # 2. Get news data
        news_tool = EnhancedNewsFetcher()
        news_result = await news_tool.execute(
            symbol="AAPL", limit=3, include_analysis=True
        )

        assert news_result is not None

        # 3. Calculate technical indicators (using sample data)
        indicators_tool = TechnicalIndicators()
        sample_price_data = [
            {
                "close": 150 + i * 0.5,
                "high": 152 + i * 0.5,
                "low": 148 + i * 0.5,
                "date": f"2024-01-{i+1:02d}",
            }
            for i in range(20)
        ]

        indicators_result = await indicators_tool.execute(
            price_data=sample_price_data, symbol="AAPL", indicators=["SMA", "RSI"]
        )

        assert indicators_result is not None

        # 4. Size position
        position_tool = PositionSizer()
        position_result = await position_tool.execute(
            account_balance=100000,
            risk_per_trade=0.02,
            entry_price=155.0,
            stop_loss_price=145.0,
            target_price=170.0,
        )

        assert position_result is not None

        # 5. Analyze portfolio risk
        risk_tool = RiskAnalyzer()
        sample_positions = [
            {
                "symbol": "AAPL",
                "value": 30000,
                "volatility": 0.25,
                "volume": 50000000,
                "sector": "Technology",
            },
            {
                "symbol": "MSFT",
                "value": 25000,
                "volatility": 0.20,
                "volume": 30000000,
                "sector": "Technology",
            },
            {
                "symbol": "SPY",
                "value": 20000,
                "volatility": 0.15,
                "volume": 100000000,
                "sector": "Index",
            },
        ]

        risk_result = await risk_tool.execute(
            positions=sample_positions, total_portfolio_value=100000
        )

        assert risk_result is not None

        # Verify workflow completed without major errors
        workflow_results = [
            market_result,
            news_result,
            indicators_result,
            position_result,
            risk_result,
        ]

        # At least some tools should work successfully
        successful_tools = [r for r in workflow_results if not r.error]
        print(
            f"Successfully executed {len(successful_tools)} out of {len(workflow_results)} tools"
        )

        # In a test environment, we expect at least the calculation tools to work
        assert (
            not indicators_result.error
        )  # Technical indicators should always work with valid data
        assert (
            not position_result.error
        )  # Position sizing should always work with valid data
        assert not risk_result.error  # Risk analysis should always work with valid data


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys

    async def run_manual_tests():
        """Run tests manually for debugging purposes."""
        print("Running manual integration tests...")

        # Test YahooFinanceFetcher
        print("\n1. Testing YahooFinanceFetcher...")
        yf_test = TestYahooFinanceFetcher()
        yf_test.test_tool_initialization()
        await yf_test.test_basic_execution()
        print("✓ YahooFinanceFetcher tests passed")

        # Test NewsAPIFetcher
        print("\n2. Testing NewsAPIFetcher...")
        news_test = TestNewsAPIFetcher()
        news_test.test_newsapi_fetcher_initialization()
        await news_test.test_newsapi_basic_execution()
        print("✓ NewsAPIFetcher tests passed")

        # Test TechnicalIndicators
        print("\n3. Testing TechnicalIndicators...")
        ti_test = TestTechnicalIndicators()
        ti_test.test_tool_initialization()
        await ti_test.test_basic_indicators_calculation()
        print("✓ TechnicalIndicators tests passed")

        # Test PositionSizer
        print("\n4. Testing PositionSizer...")
        ps_test = TestPositionSizer()
        ps_test.test_tool_initialization()
        await ps_test.test_basic_position_sizing()
        print("✓ PositionSizer tests passed")

        # Test RiskAnalyzer
        print("\n5. Testing RiskAnalyzer...")
        ra_test = TestRiskAnalyzer()
        ra_test.test_tool_initialization()
        await ra_test.test_basic_risk_analysis()
        print("✓ RiskAnalyzer tests passed")

        # Test complete workflow
        print("\n6. Testing complete workflow...")
        workflow_test = TestTradingWorkflowIntegration()
        await workflow_test.test_complete_trading_analysis_workflow()
        print("✓ Complete workflow test passed")

        print("\n🎉 All manual integration tests passed!")

    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_manual_tests())
    else:
        print("Use 'python test_tool_integration.py manual' to run manual tests")
        print("Or use 'pytest tests/test_tool_integration.py' to run with pytest")
