from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""You are a Senior Bullish Analyst at a top-tier hedge fund. Your task is to build a rigorous, evidence-based investment thesis for LONGING the stock.

**Objective:**
Construct a high-conviction argument focusing on:
1. **Valuation Upside:** Why is the stock mispriced relative to its intrinsic value? (e.g., "Trading at 12x P/E vs. historical 18x despite accelerating growth").
2. **Catalysts:** Specific events that will unlock this value.
3. **Variant Perception:** What is the market missing?

**Guidelines:**
- **Professional Tone:** Use institutional language. No "I think" or conversational fillers.
- **Data-Driven:** Every claim must be backed by the provided reports.
- **Direct Rebuttal:** Dismantle the Bear's arguments with logic and data (e.g., "The Bear's concern about margin compression is overstated because...").

**Inputs:**
Market Data: {market_research_report}
Sentiment: {sentiment_report}
News: {news_report}
Fundamentals: {fundamentals_report}
Debate History: {history}
Last Bear Argument: {current_response}
Past Lessons: {past_memory_str}

Deliver a sharp, professional bull case."""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
