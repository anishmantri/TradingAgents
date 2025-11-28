from langchain_core.prompts import ChatPromptTemplate
import json

def create_critic_agent(llm):
    def critic_node(state):
        # Extract reports
        market_report = state.get("market_report", "N/A")
        sentiment_report = state.get("sentiment_report", "N/A")
        news_report = state.get("news_report", "N/A")
        fundamentals_report = state.get("fundamentals_report", "N/A")
        
        # Extract data
        market_data = state.get("market_data", {})
        sentiment_data = state.get("sentiment_data", {})
        news_data = state.get("news_data", {})
        fundamentals_data = state.get("fundamentals_data", {})

        system_message = f"""You are a Critical Reviewer for a trading agent system. Your job is to review the reports and data provided by four analysts: Market, Social Sentiment, News, and Fundamentals.

        **Analyst Outputs:**
        
        **Market Analyst:**
        Report: {market_report}
        Data: {json.dumps(market_data, indent=2)}
        
        **Social Sentiment Analyst:**
        Report: {sentiment_report}
        Data: {json.dumps(sentiment_data, indent=2)}
        
        **News Analyst:**
        Report: {news_report}
        Data: {json.dumps(news_data, indent=2)}
        
        **Fundamentals Analyst:**
        Report: {fundamentals_report}
        Data: {json.dumps(fundamentals_data, indent=2)}
        
        **Your Task:**
        1.  **Consistency Check:** Identify any contradictions between the analysts. (e.g., Market says "Strong Uptrend" but Fundamentals says "Revenue Collapsing").
        2.  **Data Validation:** Check if the structured data matches the text report.
        3.  **Missing Information:** Identify key missing pieces of information that would be crucial for a decision.
        4.  **Confidence Assessment:** Assign a confidence score (0.0 to 1.0) to the overall analysis based on the quality and consistency of the reports.
        
        **Output Format:**
        Return a JSON object with the following structure:
        {{
            "report": "Your detailed critique report in Markdown...",
            "data": {{
                "consistency_score": 0.0 to 1.0,
                "contradictions": ["list", "of", "contradictions"],
                "missing_info": ["list", "of", "missing", "items"],
                "overall_confidence": 0.0 to 1.0
            }}
        }}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Please provide your critique.")
        ])
        
        chain = prompt | llm
        result = chain.invoke({})
        
        report = result.content
        critic_data = {}
        
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
                    critic_data = data["data"]
        except Exception:
            pass
            
        return {
            "critic_report": report,
            "critic_data": critic_data
        }

    return critic_node
