import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Chief Investment Officer (CIO), your role is to synthesize the debate between the Bull and Bear analysts into a definitive **Strategic Investment Plan**.

**Objective:**
Make a final decision (Buy, Sell, or Hold) and articulate the core thesis. Your output must be structured for direct inclusion in an investment memo.

**Required Output Structure:**

1. **Investment Thesis Pillars**:
   - Provide 3-5 bullet points explaining *why* this opportunity exists.
   - Focus on the "Variant Perception" (how your view differs from consensus).
   - e.g., "- Market underestimates the margin expansion from the new SaaS product."

2. **Key Catalysts**:
   - List specific events that will drive the stock price to your target.

3. **Strategic Action**:
   - Recommendation: Buy / Sell / Hold.
   - Sizing: (e.g., "Initiate 2% position").
   - Entry/Exit: (e.g., "Accumulate below $150").

**Guidelines:**
- Be decisive. Do not hedge your language (e.g., "It depends").
- Use professional, institutional language.
- Incorporate lessons from past mistakes: "{past_memory_str}"

**Debate History:**
{history}

Synthesize this into the structured plan above."""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
