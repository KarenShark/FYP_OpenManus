import asyncio
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class RiskWarning(BaseModel):
    """Represents a risk warning or alert."""

    level: str = Field(description="Warning level: LOW, MEDIUM, HIGH, CRITICAL")
    message: str = Field(description="Warning message")
    recommendation: str = Field(description="Recommended action")


class PositionSizeResult(BaseModel):
    """Result of position sizing calculation."""

    recommended_position_size: float = Field(
        description="Recommended number of shares/units"
    )
    recommended_position_value: float = Field(
        description="Total value of recommended position"
    )
    actual_risk_percentage: float = Field(
        description="Actual risk as percentage of account"
    )
    actual_risk_amount: float = Field(description="Actual risk amount in currency")
    max_loss_amount: float = Field(description="Maximum potential loss")
    entry_price: float = Field(description="Entry price used in calculation")
    stop_loss_price: float = Field(description="Stop loss price used")
    risk_reward_ratio: Optional[float] = Field(
        default=None, description="Risk to reward ratio"
    )


class RiskAnalysisResult(BaseModel):
    """Result of comprehensive risk analysis."""

    overall_risk_score: float = Field(
        description="Overall risk score from 0 (low) to 10 (high)"
    )
    portfolio_risk_percentage: float = Field(description="Portfolio risk as percentage")
    position_correlation_risk: float = Field(
        description="Risk from correlated positions"
    )
    volatility_risk: float = Field(description="Risk from price volatility")
    liquidity_risk: float = Field(description="Risk from low liquidity")
    concentration_risk: float = Field(description="Risk from position concentration")


class PositionSizerResponse(ToolResult):
    """Structured response from position sizing calculation."""

    position_result: Optional[PositionSizeResult] = Field(default=None)
    risk_warnings: List[RiskWarning] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Populate output field after model initialization."""
        if self.error:
            return

        if not self.position_result:
            self.output = "Position sizing calculation failed."
            return

        result = self.position_result

        output_lines = [
            "Position Sizing Analysis:",
            f"Entry Price: ${result.entry_price:.2f}",
            f"Stop Loss Price: ${result.stop_loss_price:.2f}",
            f"Risk per Share: ${abs(result.entry_price - result.stop_loss_price):.2f}",
            "",
            "Recommended Position:",
            f"  Size: {result.recommended_position_size:.0f} shares",
            f"  Value: ${result.recommended_position_value:,.2f}",
            f"  Risk Amount: ${result.actual_risk_amount:,.2f}",
            f"  Risk Percentage: {result.actual_risk_percentage:.2f}%",
            f"  Max Loss: ${result.max_loss_amount:,.2f}",
        ]

        if result.risk_reward_ratio:
            output_lines.append(
                f"  Risk/Reward Ratio: 1:{result.risk_reward_ratio:.2f}"
            )

        # Add risk warnings
        if self.risk_warnings:
            output_lines.extend(["", "Risk Warnings:"])
            for warning in self.risk_warnings:
                output_lines.append(f"  {warning.level}: {warning.message}")
                output_lines.append(f"    Action: {warning.recommendation}")

        self.output = "\n".join(output_lines)


class RiskAnalyzerResponse(ToolResult):
    """Structured response from risk analysis."""

    analysis_result: Optional[RiskAnalysisResult] = Field(default=None)
    risk_warnings: List[RiskWarning] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Populate output field after model initialization."""
        if self.error:
            return

        if not self.analysis_result:
            self.output = "Risk analysis failed."
            return

        result = self.analysis_result

        output_lines = [
            "Portfolio Risk Analysis:",
            f"Overall Risk Score: {result.overall_risk_score:.1f}/10",
            "",
            "Risk Breakdown:",
            f"  Portfolio Risk: {result.portfolio_risk_percentage:.2f}%",
            f"  Position Correlation: {result.position_correlation_risk:.2f}/10",
            f"  Volatility Risk: {result.volatility_risk:.2f}/10",
            f"  Liquidity Risk: {result.liquidity_risk:.2f}/10",
            f"  Concentration Risk: {result.concentration_risk:.2f}/10",
        ]

        # Risk level interpretation
        if result.overall_risk_score >= 8:
            risk_level = "CRITICAL"
        elif result.overall_risk_score >= 6:
            risk_level = "HIGH"
        elif result.overall_risk_score >= 4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        output_lines.extend(["", f"Risk Level: {risk_level}"])

        # Add risk warnings
        if self.risk_warnings:
            output_lines.extend(["", "Risk Warnings:"])
            for warning in self.risk_warnings:
                output_lines.append(f"  {warning.level}: {warning.message}")
                output_lines.append(f"    Action: {warning.recommendation}")

        self.output = "\n".join(output_lines)


class PositionSizer(BaseTool):
    """Calculate optimal position size based on risk management principles."""

    name: str = "position_sizer"
    description: str = """Calculate optimal position size for a trade based on account balance, risk tolerance,
    entry price, and stop loss. Implements proper risk management to preserve capital."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "account_balance": {
                "type": "number",
                "description": "(required) Total account balance in currency units.",
            },
            "risk_per_trade": {
                "type": "number",
                "description": "(required) Risk percentage per trade (e.g., 0.02 for 2%).",
            },
            "entry_price": {
                "type": "number",
                "description": "(required) Planned entry price for the position.",
            },
            "stop_loss_price": {
                "type": "number",
                "description": "(required) Stop loss price for the position.",
            },
            "target_price": {
                "type": "number",
                "description": "(optional) Target price for calculating risk/reward ratio.",
            },
            "current_portfolio_value": {
                "type": "number",
                "description": "(optional) Current portfolio value for concentration analysis.",
            },
        },
        "required": [
            "account_balance",
            "risk_per_trade",
            "entry_price",
            "stop_loss_price",
        ],
    }

    async def execute(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
        target_price: Optional[float] = None,
        current_portfolio_value: Optional[float] = None,
    ) -> PositionSizerResponse:
        """
        Calculate optimal position size for a trade.

        Args:
            account_balance: Total account balance
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            target_price: Target price for risk/reward calculation
            current_portfolio_value: Current portfolio value

        Returns:
            PositionSizerResponse with position sizing recommendations
        """
        try:
            logger.info(
                f"Calculating position size - Balance: ${account_balance}, Risk: {risk_per_trade*100}%"
            )

            # Validate inputs
            if account_balance <= 0:
                return PositionSizerResponse(error="Account balance must be positive")

            if risk_per_trade <= 0 or risk_per_trade > 1:
                return PositionSizerResponse(
                    error="Risk per trade must be between 0 and 1 (0% to 100%)"
                )

            if entry_price <= 0 or stop_loss_price <= 0:
                return PositionSizerResponse(
                    error="Entry price and stop loss price must be positive"
                )

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)

            if risk_per_share == 0:
                return PositionSizerResponse(
                    error="Entry price and stop loss price cannot be the same"
                )

            # Calculate maximum risk amount
            max_risk_amount = account_balance * risk_per_trade

            # Calculate position size
            position_size = max_risk_amount / risk_per_share
            position_value = position_size * entry_price

            # Calculate actual risk metrics
            actual_risk_amount = position_size * risk_per_share
            actual_risk_percentage = (actual_risk_amount / account_balance) * 100

            # Calculate risk/reward ratio if target price provided
            risk_reward_ratio = None
            if target_price:
                potential_profit_per_share = abs(target_price - entry_price)
                if risk_per_share > 0:
                    risk_reward_ratio = potential_profit_per_share / risk_per_share

            # Create position result
            position_result = PositionSizeResult(
                recommended_position_size=position_size,
                recommended_position_value=position_value,
                actual_risk_percentage=actual_risk_percentage,
                actual_risk_amount=actual_risk_amount,
                max_loss_amount=actual_risk_amount,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_reward_ratio=risk_reward_ratio,
            )

            # Generate risk warnings
            warnings = await self._generate_position_warnings(
                position_result, account_balance, current_portfolio_value
            )

            logger.info(
                f"Position sizing complete - Size: {position_size:.0f} shares, Value: ${position_value:,.2f}"
            )

            return PositionSizerResponse(
                position_result=position_result, risk_warnings=warnings
            )

        except Exception as e:
            error_msg = f"Error calculating position size: {str(e)}"
            logger.error(error_msg)
            return PositionSizerResponse(error=error_msg)

    async def _generate_position_warnings(
        self,
        result: PositionSizeResult,
        account_balance: float,
        portfolio_value: Optional[float],
    ) -> List[RiskWarning]:
        """Generate risk warnings based on position sizing results."""
        warnings = []

        # High risk percentage warning
        if result.actual_risk_percentage > 5:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message=f"Risk per trade is {result.actual_risk_percentage:.2f}%, exceeding recommended 2-3%",
                    recommendation="Consider reducing position size or tightening stop loss",
                )
            )
        elif result.actual_risk_percentage > 3:
            warnings.append(
                RiskWarning(
                    level="MEDIUM",
                    message=f"Risk per trade is {result.actual_risk_percentage:.2f}%, above conservative 2%",
                    recommendation="Monitor position closely and consider risk reduction",
                )
            )

        # Large position value warning
        position_percentage = (
            result.recommended_position_value / account_balance
        ) * 100
        if position_percentage > 50:
            warnings.append(
                RiskWarning(
                    level="CRITICAL",
                    message=f"Position value is {position_percentage:.1f}% of account balance",
                    recommendation="Significantly reduce position size to avoid over-concentration",
                )
            )
        elif position_percentage > 25:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message=f"Position value is {position_percentage:.1f}% of account balance",
                    recommendation="Consider reducing position size for better diversification",
                )
            )

        # Portfolio concentration warning
        if portfolio_value:
            portfolio_percentage = (
                result.recommended_position_value / portfolio_value
            ) * 100
            if portfolio_percentage > 20:
                warnings.append(
                    RiskWarning(
                        level="HIGH",
                        message=f"Position would be {portfolio_percentage:.1f}% of total portfolio",
                        recommendation="Reduce position size to maintain portfolio diversification",
                    )
                )

        # Poor risk/reward ratio warning
        if result.risk_reward_ratio and result.risk_reward_ratio < 1.5:
            warnings.append(
                RiskWarning(
                    level="MEDIUM",
                    message=f"Risk/reward ratio is {result.risk_reward_ratio:.2f}, below recommended 1:2",
                    recommendation="Consider better entry/exit points or skip this trade",
                )
            )

        # Minimum position size warning
        if result.recommended_position_size < 1:
            warnings.append(
                RiskWarning(
                    level="LOW",
                    message="Position size is less than 1 share",
                    recommendation="Consider increasing risk tolerance or finding lower-priced assets",
                )
            )

        return warnings


class RiskAnalyzer(BaseTool):
    """Analyze portfolio risk across multiple dimensions."""

    name: str = "risk_analyzer"
    description: str = """Perform comprehensive portfolio risk analysis including correlation,
    volatility, concentration, and liquidity risks. Provides overall risk score and recommendations."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "positions": {
                "type": "array",
                "description": "(required) Array of current positions with symbol, value, and volatility.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "value": {"type": "number"},
                        "volatility": {"type": "number"},
                        "volume": {"type": "number"},
                        "sector": {"type": "string"},
                    },
                    "required": ["symbol", "value"],
                },
            },
            "total_portfolio_value": {
                "type": "number",
                "description": "(required) Total portfolio value.",
            },
            "risk_free_rate": {
                "type": "number",
                "description": "(optional) Risk-free rate for calculations. Default is 0.02 (2%).",
                "default": 0.02,
            },
        },
        "required": ["positions", "total_portfolio_value"],
    }

    async def execute(
        self,
        positions: List[Dict[str, Any]],
        total_portfolio_value: float,
        risk_free_rate: float = 0.02,
    ) -> RiskAnalyzerResponse:
        """
        Perform comprehensive portfolio risk analysis.

        Args:
            positions: List of portfolio positions
            total_portfolio_value: Total portfolio value
            risk_free_rate: Risk-free rate for calculations

        Returns:
            RiskAnalyzerResponse with comprehensive risk analysis
        """
        try:
            logger.info(
                f"Analyzing portfolio risk - {len(positions)} positions, ${total_portfolio_value:,.2f} total value"
            )

            if not positions:
                return RiskAnalyzerResponse(error="No positions provided for analysis")

            if total_portfolio_value <= 0:
                return RiskAnalyzerResponse(
                    error="Total portfolio value must be positive"
                )

            # Calculate individual risk components
            concentration_risk = await self._calculate_concentration_risk(
                positions, total_portfolio_value
            )
            volatility_risk = await self._calculate_volatility_risk(positions)
            correlation_risk = await self._calculate_correlation_risk(positions)
            liquidity_risk = await self._calculate_liquidity_risk(positions)

            # Calculate portfolio risk percentage
            total_position_value = sum(pos.get("value", 0) for pos in positions)
            portfolio_risk_percentage = (
                total_position_value / total_portfolio_value
            ) * 100

            # Calculate overall risk score (0-10 scale)
            overall_risk_score = (
                concentration_risk * 0.3
                + volatility_risk * 0.25
                + correlation_risk * 0.25
                + liquidity_risk * 0.2
            )

            # Create analysis result
            analysis_result = RiskAnalysisResult(
                overall_risk_score=overall_risk_score,
                portfolio_risk_percentage=portfolio_risk_percentage,
                position_correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
            )

            # Generate risk warnings
            warnings = await self._generate_risk_warnings(
                analysis_result, positions, total_portfolio_value
            )

            logger.info(
                f"Risk analysis complete - Overall score: {overall_risk_score:.1f}/10"
            )

            return RiskAnalyzerResponse(
                analysis_result=analysis_result, risk_warnings=warnings
            )

        except Exception as e:
            error_msg = f"Error analyzing portfolio risk: {str(e)}"
            logger.error(error_msg)
            return RiskAnalyzerResponse(error=error_msg)

    async def _calculate_concentration_risk(
        self, positions: List[Dict], total_value: float
    ) -> float:
        """Calculate concentration risk based on position sizes."""
        if not positions:
            return 0.0

        # Calculate position weights
        weights = []
        for pos in positions:
            weight = pos.get("value", 0) / total_value
            weights.append(weight)

        # Calculate concentration using Herfindahl-Hirschman Index
        hhi = sum(w * w for w in weights)

        # Convert to 0-10 scale (higher HHI = higher concentration risk)
        # HHI ranges from 1/n (perfectly diversified) to 1 (single position)
        max_hhi = 1.0
        min_hhi = 1.0 / len(positions) if len(positions) > 0 else 1.0

        normalized_hhi = (
            (hhi - min_hhi) / (max_hhi - min_hhi) if max_hhi > min_hhi else 0.0
        )
        return min(normalized_hhi * 10, 10.0)

    async def _calculate_volatility_risk(self, positions: List[Dict]) -> float:
        """Calculate volatility risk based on position volatilities."""
        if not positions:
            return 0.0

        volatilities = []
        weights = []
        total_value = sum(pos.get("value", 0) for pos in positions)

        for pos in positions:
            vol = pos.get("volatility", 0.2)  # Default 20% volatility
            weight = pos.get("value", 0) / total_value if total_value > 0 else 0
            volatilities.append(vol)
            weights.append(weight)

        # Calculate weighted average volatility
        if weights:
            avg_volatility = sum(v * w for v, w in zip(volatilities, weights))
        else:
            avg_volatility = 0.2

        # Convert to 0-10 scale (20% vol = 2, 100% vol = 10)
        risk_score = min(avg_volatility * 10, 10.0)
        return risk_score

    async def _calculate_correlation_risk(self, positions: List[Dict]) -> float:
        """Calculate correlation risk based on sector/asset diversification."""
        if len(positions) <= 1:
            return 10.0  # Maximum risk for single position

        # Group by sector
        sectors = {}
        total_value = sum(pos.get("value", 0) for pos in positions)

        for pos in positions:
            sector = pos.get("sector", "Unknown")
            value = pos.get("value", 0)
            if sector in sectors:
                sectors[sector] += value
            else:
                sectors[sector] = value

        # Calculate sector concentration
        sector_weights = (
            [v / total_value for v in sectors.values()] if total_value > 0 else []
        )

        if sector_weights:
            # Calculate Herfindahl index for sectors
            sector_hhi = sum(w * w for w in sector_weights)
            # Convert to 0-10 scale
            max_sector_hhi = 1.0
            min_sector_hhi = 1.0 / len(sectors) if len(sectors) > 0 else 1.0
            normalized_hhi = (
                (sector_hhi - min_sector_hhi) / (max_sector_hhi - min_sector_hhi)
                if max_sector_hhi > min_sector_hhi
                else 0.0
            )
            return min(normalized_hhi * 10, 10.0)

        return 5.0  # Default medium risk

    async def _calculate_liquidity_risk(self, positions: List[Dict]) -> float:
        """Calculate liquidity risk based on trading volumes."""
        if not positions:
            return 0.0

        liquidity_scores = []
        weights = []
        total_value = sum(pos.get("value", 0) for pos in positions)

        for pos in positions:
            volume = pos.get("volume", 1000000)  # Default 1M volume
            weight = pos.get("value", 0) / total_value if total_value > 0 else 0

            # Convert volume to liquidity score (higher volume = lower risk)
            if volume >= 10000000:  # Very liquid
                liquidity_score = 1.0
            elif volume >= 1000000:  # Liquid
                liquidity_score = 3.0
            elif volume >= 100000:  # Moderately liquid
                liquidity_score = 5.0
            elif volume >= 10000:  # Low liquidity
                liquidity_score = 7.0
            else:  # Very low liquidity
                liquidity_score = 9.0

            liquidity_scores.append(liquidity_score)
            weights.append(weight)

        # Calculate weighted average liquidity risk
        if weights:
            avg_liquidity_risk = sum(l * w for l, w in zip(liquidity_scores, weights))
        else:
            avg_liquidity_risk = 5.0

        return min(avg_liquidity_risk, 10.0)

    async def _generate_risk_warnings(
        self, analysis: RiskAnalysisResult, positions: List[Dict], total_value: float
    ) -> List[RiskWarning]:
        """Generate risk warnings based on analysis results."""
        warnings = []

        # Overall risk warning
        if analysis.overall_risk_score >= 8:
            warnings.append(
                RiskWarning(
                    level="CRITICAL",
                    message=f"Overall portfolio risk is very high ({analysis.overall_risk_score:.1f}/10)",
                    recommendation="Immediately review and reduce portfolio risk through diversification",
                )
            )
        elif analysis.overall_risk_score >= 6:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message=f"Overall portfolio risk is high ({analysis.overall_risk_score:.1f}/10)",
                    recommendation="Consider reducing risk through better diversification",
                )
            )

        # Concentration risk warning
        if analysis.concentration_risk >= 7:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message="Portfolio shows high concentration risk",
                    recommendation="Diversify across more positions to reduce concentration",
                )
            )

        # Volatility risk warning
        if analysis.volatility_risk >= 7:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message="Portfolio shows high volatility risk",
                    recommendation="Consider adding more stable, lower-volatility assets",
                )
            )

        # Correlation risk warning
        if analysis.position_correlation_risk >= 7:
            warnings.append(
                RiskWarning(
                    level="HIGH",
                    message="Positions may be highly correlated",
                    recommendation="Diversify across different sectors and asset classes",
                )
            )

        # Liquidity risk warning
        if analysis.liquidity_risk >= 7:
            warnings.append(
                RiskWarning(
                    level="MEDIUM",
                    message="Some positions may have liquidity concerns",
                    recommendation="Ensure adequate liquidity for position management",
                )
            )

        # Large position warnings
        for pos in positions:
            pos_percentage = (pos.get("value", 0) / total_value) * 100
            if pos_percentage > 25:
                warnings.append(
                    RiskWarning(
                        level="HIGH",
                        message=f"Position {pos.get('symbol', 'Unknown')} is {pos_percentage:.1f}% of portfolio",
                        recommendation="Consider reducing this position to improve diversification",
                    )
                )

        return warnings


if __name__ == "__main__":
    # Test PositionSizer
    sizer = PositionSizer()
    size_result = asyncio.run(
        sizer.execute(
            account_balance=100000,
            risk_per_trade=0.02,
            entry_price=150.0,
            stop_loss_price=140.0,
            target_price=170.0,
        )
    )
    print("Position Sizer Result:")
    print(size_result.output if size_result.output else size_result.error)

    print("\n" + "=" * 50 + "\n")

    # Test RiskAnalyzer
    analyzer = RiskAnalyzer()
    sample_positions = [
        {
            "symbol": "AAPL",
            "value": 25000,
            "volatility": 0.25,
            "volume": 50000000,
            "sector": "Technology",
        },
        {
            "symbol": "TSLA",
            "value": 30000,
            "volatility": 0.35,
            "volume": 25000000,
            "sector": "Technology",
        },
        {
            "symbol": "JPM",
            "value": 20000,
            "volatility": 0.20,
            "volume": 15000000,
            "sector": "Finance",
        },
        {
            "symbol": "SPY",
            "value": 15000,
            "volatility": 0.15,
            "volume": 100000000,
            "sector": "Index",
        },
    ]

    risk_result = asyncio.run(
        analyzer.execute(positions=sample_positions, total_portfolio_value=100000)
    )
    print("Risk Analyzer Result:")
    print(risk_result.output if risk_result.output else risk_result.error)
