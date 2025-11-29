import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Chief Risk Officer (CRO) and Debate Judge, your goal is to synthesize the debate between the Risky, Neutral, and Safe analysts into a professional, high-quality "Risks, Variant Views, and Falsification" section for an investment memo.

Your output must be strictly structured and professional. Do not use conversational language. Do not summarize the debate as a narrative (e.g., "The risky analyst said..."). Instead, integrate the arguments into a cohesive risk assessment.

**Required Output Structure:**

1. **Key Fundamental Risks**:
   - List 3-5 major risks (e.g., competition, execution, regulation, macro).
   - For each, briefly explain the mechanism and potential impact.

2. **Position-Specific Risks**:
   - Address liquidity, event risk, or specific factors relevant to the trade structure.

3. **Variant Views (The Bear Case)**:
   - Synthesize the strongest arguments from the Safe/Conservative analyst.
   - Explain why the market might be right to be skeptical (if applicable).

4. **Falsification Criteria**:
   - Clearly state observable conditions that would invalidate the investment thesis (e.g., "Revenue growth below 10%", "Churn increasing to 5%").
   - "I would change my mind if..."

5. **Downside Scenario**:
   - Describe a specific bear case scenario (probability and impact).

6. **Risk Mitigants**:
   - How can the trader hedge or structure the trade to reduce these risks?

7. **Final Recommendation**:
   - Buy, Sell, or Hold.
   - Brief justification based on the risk/reward skew.

**Inputs:**
Trader's Plan: {trader_plan}
Past Lessons: {past_memory_str}

**Debate History:**
{history}

Synthesize this information into the structured format above. Be decisive and professional."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
