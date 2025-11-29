from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.dataflows.config import get_config
from tradingagents.utils.timeframes import describe_window, lookback_start


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        config = get_config()
        lookback_days = config.get("analysis_window_days", 7)
        window_desc = describe_window(lookback_days)
        window_start = lookback_start(current_date, lookback_days)

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            f"""You are a quantitative trading assistant tasked with analyzing financial markets over the {window_desc}. Focus on price action between {window_start} and {current_date}. Your role is to select the **most relevant quantitative metrics** for the current market regime. Choose up to **8 indicators** that provide orthogonal (non-redundant) signals.

Available Quantitative Metrics:

Trend & Regime:
- market_regime: Automated classification (Trending Bull/Bear, Mean Reverting, Transition). Usage: PRIMARY filter. If Trending -> use Trend Following tools. If Ranging -> use Mean Reversion tools.
- close_50_sma: Medium-term baseline. Usage: Assess regime bias (Bullish > SMA, Bearish < SMA).
- close_200_sma: Structural baseline. Usage: Institutional support/resistance and long-term trend definition.

Momentum (Volatility-Adjusted):
- rsi_vol_scaled: Volatility-Scaled RSI (Z-Score). Usage: Identifies statistical extremes (> +2σ Overbought, < -2σ Oversold) adjusted for current volatility. superior to fixed 70/30 thresholds.
- macd_impulse: Impulse MACD. Usage: Detects high-confidence momentum shifts where momentum breaks out of its own volatility bands. Filters sideways noise.

Flow & Volume:
- vpt: Volume Price Trend. Usage: Institutional flow proxy. Confirm trend strength (Price up + VPT up) or spot divergence (Price up + VPT flat/down = Distribution).

Volatility:
- boll: Bollinger Bands (2 Std Dev). Usage: Volatility regime (Squeeze vs Expansion) and mean reversion boundaries.
- atr: Absolute Volatility. Usage: Gauge market "temperature" and risk level.

INSTRUCTIONS:
1. **Identify Regime First**: Use `market_regime` and SMAs to determine if the market is Trending or Ranging.
2. **Select Appropriate Tools**:
   - If Trending: Prioritize `vpt` (flow confirmation) and `close_10_ema` (trailing stop).
   - If Ranging/Mean Reverting: Prioritize `rsi_vol_scaled` (statistical extremes) and `boll` (bands).
   - If Transition/Breakout: Look for `macd_impulse` and Volatility Expansion (`boll`).
3. **Professional Analysis**:
   - AVOID retail clichés like "overbought/oversold" without statistical context. Use "statistical extension" or "mean reversion probability".
   - Focus on **Confluence**: Do Flow (`vpt`) and Momentum (`macd_impulse`) agree?
   - Assess **Risk/Reward**: Is volatility (`atr`) expanding or contracting?
   - Provide a nuanced, institutional-grade assessment of the market structure.

Select indicators that provide diverse and complementary information. Avoid redundancy. Call `get_stock_data` first, then `get_indicators` with the specific indicator names."""
            + """ 
            
            CRITICAL OUTPUT FORMAT:
            You must return your final response as a JSON object with the following structure:
            {{
                "report": "Your detailed Markdown report here...",
                "data": {{
                    "signal": "bullish" | "bearish" | "neutral",
                    "confidence": 0.0 to 1.0,
                    "key_metrics": {{ "metric_name": "value", ... }}
                }}
            }}
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = result.content
        market_data = {}

        if len(result.tool_calls) == 0:
            # Try to parse JSON output
            try:
                content = result.content
                json_str = content
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                
                data = json.loads(json_str.strip())
                if isinstance(data, dict):
                    if "report" in data:
                        report = data["report"]
                    if "data" in data:
                        market_data = data["data"]
            except Exception:
                # Fallback to raw content if parsing fails
                pass
       
        return {
            "messages": [result],
            "market_report": report,
            "market_data": market_data,
        }

    return market_analyst_node
