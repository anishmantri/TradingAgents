from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_sentiment, get_insider_transactions
from tradingagents.dataflows.config import get_config
from tradingagents.utils.timeframes import describe_window, lookback_start


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]
        config = get_config()
        lookback_days = config.get("analysis_window_days", 7)
        window_desc = describe_window(lookback_days)
        window_start = lookback_start(current_date, lookback_days)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            f"You are a Fundamental Equity Analyst. Your goal is to dissect the company's financial health and business model for {window_desc} (covering {window_start} to {current_date})."
            " Go beyond basic ratios. Analyze:"
            "\n1. **Quality of Earnings:** Is growth organic or driven by one-offs/accounting tricks?"
            "\n2. **Capital Allocation:** ROIC trends, buybacks vs. capex, management incentives."
            "\n3. **Moat Durability:** Pricing power, margin stability, competitive erosion."
            "\n\nProduce a professional 'Company Overview' and 'Financial Analysis' report."
            + """ 
            
            CRITICAL OUTPUT FORMAT:
            You must return your final response as a JSON object with the following structure:
            {{
                "report": "Your detailed Markdown report here...",
                "data": {{
                    "signal": "bullish" | "bearish" | "neutral",
                    "confidence": 0.0 to 1.0,
                    "financial_health": "strong" | "weak" | "stable",
                    "key_ratios": {{ "ratio_name": "value", ... }}
                }}
            }}
            """
            + " Use the available tools: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`.",
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
        fundamentals_data = {}

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
                        fundamentals_data = data["data"]
            except Exception:
                # Fallback to raw content if parsing fails
                pass

        return {
            "messages": [result],
            "fundamentals_report": report,
            "fundamentals_data": fundamentals_data,
        }

    return fundamentals_analyst_node
