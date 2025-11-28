from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config
from tradingagents.utils.timeframes import describe_window, lookback_start


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        config = get_config()
        lookback_days = config.get("analysis_window_days", 7)
        window_desc = describe_window(lookback_days)
        window_start = lookback_start(current_date, lookback_days)

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            f"You are a news researcher tasked with analyzing trading-relevant developments over the {window_desc}. Focus on news released between {window_start} and {current_date}. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ 
            
            CRITICAL OUTPUT FORMAT:
            You must return your final response as a JSON object with the following structure:
            {
                "report": "Your detailed Markdown report here...",
                "data": {
                    "signal": "bullish" | "bearish" | "neutral",
                    "confidence": 0.0 to 1.0,
                    "impact_score": 0.0 to 10.0,
                    "key_events": ["event1", "event2"]
                }
            }
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
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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
        news_data = {}

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
                        news_data = data["data"]
            except Exception:
                # Fallback to raw content if parsing fails
                pass

        return {
            "messages": [result],
            "news_report": report,
            "news_data": news_data,
        }

    return news_analyst_node
