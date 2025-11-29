import chromadb
from chromadb.config import Settings
from openai import OpenAI

from tradingagents.utils.provider import get_provider_api_key


class FinancialSituationMemory:
    def __init__(self, name, config):
        provider_key = get_provider_api_key(config["llm_provider"])
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"
        client_kwargs = {"base_url": config["backend_url"]}
        if provider_key:
            client_kwargs["api_key"] = provider_key
        self.client = OpenAI(**client_kwargs)
        self.summary_model = config.get("quick_think_llm", "gpt-4o-mini")
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def _summarize_text(self, text):
        """Summarize long text to fit within embedding limits"""
        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Summarize the following market report into a dense, high-signal overview capturing key trends, risks, and signals. Keep it under 1000 words."
                    },
                    {"role": "user", "content": text[:50000]}  # Hard cap for summarization input
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Summarization failed: {e}. Falling back to truncation.")
            return text[:24000]

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        # Check if text needs summarization (approx 6000 tokens / 24k chars)
        text_to_embed = text
        if len(text) > 24000:
            text_to_embed = self._summarize_text(text)
        
        # Final safety truncation just in case summary is still too long or failed
        text_to_embed = text_to_embed[:30000]
        
        response = self.client.embeddings.create(
            model=self.embedding, input=text_to_embed
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
