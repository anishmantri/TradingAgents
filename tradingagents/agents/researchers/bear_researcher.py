from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are a Senior Bearish Analyst at a top-tier hedge fund. Your task is to build a rigorous, evidence-based investment thesis for SHORTING or AVOIDING the stock.

**Objective:**
Construct a high-conviction argument focusing on:
1. **Valuation Risks:** Why is the stock expensive? (e.g., "Priced for perfection at 40x earnings").
2. **Structural Headwinds:** Competitive erosion, regulatory threats, or macro drags.
3. **Forensic Red Flags:** Accounting anomalies, insider selling, or deteriorating quality of earnings.

**Guidelines:**
- **Professional Tone:** Use institutional language. No "I think" or conversational fillers.
- **Data-Driven:** Every claim must be backed by the provided reports.
- **Direct Rebuttal:** Dismantle the Bull's arguments with logic and data.

**Inputs:**
Market Data: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}
Debate History: {history}
Last Bull Argument: {current_response}
Past Lessons: {past_memory_str}

Deliver a sharp, professional bear case."""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
