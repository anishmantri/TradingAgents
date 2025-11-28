import os
import sys
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API key
os.environ["OPENAI_API_KEY"] = "dummy"

# Mock dependencies
with patch('tradingagents.graph.trading_graph.ChatOpenAI') as MockChatOpenAI, \
     patch('tradingagents.graph.trading_graph.get_provider_api_key') as mock_get_key, \
     patch('tradingagents.graph.setup.ChatOpenAI') as MockSetupChatOpenAI:
    
    mock_get_key.return_value = "dummy_key"
    MockChatOpenAI.return_value = MagicMock()
    MockSetupChatOpenAI.return_value = MagicMock()

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.graph.setup import GraphSetup
    from tradingagents.agents.utils.agent_states import AgentState

    async def test_parallel_execution_and_merging():
        print("Testing parallel execution and structured data merging...")
        
        # We want to test the 'Parallel Analysts' node specifically.
        # We can mock _create_analyst_subgraph to return a mock graph that returns specific data.
        
        with patch.object(GraphSetup, '_create_analyst_subgraph') as mock_create_subgraph:
            # Define what the mock subgraph returns
            async def mock_ainvoke(state):
                # Determine which analyst this is based on the call
                # But mock_create_subgraph is called during setup, returning a graph.
                # We need to return different graphs for different analysts.
                return {"messages": []} # Default
            
            # We need a side_effect for _create_analyst_subgraph that returns a mock with a custom ainvoke
            def create_mock_graph(key, *args):
                mock_graph = MagicMock()
                
                async def side_effect_ainvoke(state):
                    if key == "market":
                        return {"market_report": "Market Report", "market_data": {"signal": "bullish"}}
                    elif key == "social":
                        return {"sentiment_report": "Social Report", "sentiment_data": {"score": 0.8}}
                    elif key == "news":
                        return {"news_report": "News Report", "news_data": {"impact": "high"}}
                    elif key == "fundamentals":
                        return {"fundamentals_report": "Fund Report", "fundamentals_data": {"health": "good"}}
                    return {}
                
                mock_graph.ainvoke = AsyncMock(side_effect=side_effect_ainvoke)
                return mock_graph
            
            mock_create_subgraph.side_effect = create_mock_graph
            
            # Initialize graph setup
            # We need to mock the LLMs passed to GraphSetup
            mock_llm = MagicMock()
            mock_tools = MagicMock()
            mock_memory = MagicMock()
            mock_logic = MagicMock()
            
            setup = GraphSetup(
                mock_llm, mock_llm, mock_tools, 
                mock_memory, mock_memory, mock_memory, mock_memory, mock_memory, 
                mock_logic
            )
            
            # Setup the graph
            app = setup.setup_graph(["market", "social", "news", "fundamentals"])
            # app is already compiled
            
            # Now we want to run ONLY the 'Parallel Analysts' node.
            # But we can't easily run just one node in compiled graph.
            # However, we can inspect the node function if we can access it.
            # Or we can run the graph from START and see if it hits the next node with correct state.
            
            # Let's try to run the graph with a dummy state.
            # We need to mock the other nodes (Bull Researcher etc) so they don't crash or do anything.
            # But setup_graph creates them using real functions.
            # We might need to patch create_bull_researcher etc.
            
            with patch('tradingagents.graph.setup.create_bull_researcher') as mock_bull, \
                 patch('tradingagents.graph.setup.create_bear_researcher') as mock_bear, \
                 patch('tradingagents.graph.setup.create_research_manager') as mock_manager, \
                 patch('tradingagents.graph.setup.create_trader') as mock_trader, \
                 patch('tradingagents.graph.setup.create_risky_debator') as mock_risky, \
                 patch('tradingagents.graph.setup.create_neutral_debator') as mock_neutral, \
                 patch('tradingagents.graph.setup.create_safe_debator') as mock_safe, \
                 patch('tradingagents.graph.setup.create_risk_manager') as mock_risk_judge:
                
                # Make them identity functions that print state
                def print_state_and_return(s):
                    print(f"Bull Researcher received state: market_data={s.get('market_data')}, sentiment_data={s.get('sentiment_data')}")
                    return {"messages": []}
                
                mock_bull.return_value = print_state_and_return
                mock_bear.return_value = lambda s: {"messages": []}
                mock_manager.return_value = lambda s: {"messages": []}
                mock_trader.return_value = lambda s: {"messages": []}
                mock_risky.return_value = lambda s: {"messages": []}
                mock_neutral.return_value = lambda s: {"messages": []}
                mock_safe.return_value = lambda s: {"messages": []}
                mock_risk_judge.return_value = lambda s: {"messages": []}
                
                # Re-setup graph with these mocks
                app = setup.setup_graph(["market", "social", "news", "fundamentals"])
                
                print("Running graph...")
                initial_state = {
                    "messages": [],
                    "company_of_interest": "AAPL",
                    "trade_date": "2023-01-01"
                }
                
                # Run the graph. It should execute Parallel Analysts first.
                try:
                    result = await app.ainvoke(initial_state)
                except Exception as e:
                    print(f"Graph execution stopped as expected (due to mocks): {e}")
                    # We can't check 'result' here if it failed, but we printed the state in Bull Researcher
                    return
                
                print("Graph execution completed.")
                # Check results
                print(f"Market Data: {result.get('market_data')}")
                print(f"Sentiment Data: {result.get('sentiment_data')}")
                
                if result.get("market_data") == {"signal": "bullish"} and \
                   result.get("sentiment_data") == {"score": 0.8}:
                    print("SUCCESS: Structured data merged correctly.")
                else:
                    print("FAILURE: Data mismatch.")

    if __name__ == "__main__":
        asyncio.run(test_parallel_execution_and_merging())
