# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic

    def _create_analyst_subgraph(self, analyst_type, analyst_node, tools_node, condition_method):
        """Helper to create a subgraph for a single analyst."""
        workflow = StateGraph(AgentState)
        
        # Node names
        analyst_name = f"{analyst_type.capitalize()} Analyst"
        tools_name = f"tools_{analyst_type}"
        clear_name = f"Msg Clear {analyst_type.capitalize()}"
        
        workflow.add_node(analyst_name, analyst_node)
        workflow.add_node(tools_name, tools_node)
        
        workflow.add_edge(START, analyst_name)
        workflow.add_edge(tools_name, analyst_name)
        
        # Map the conditional outputs
        # "tools_..." -> tools node
        # "Msg Clear ..." -> END (we don't need the clear node in the subgraph, just end)
        workflow.add_conditional_edges(
            analyst_name,
            condition_method,
            {
                tools_name: tools_name,
                clear_name: END
            }
        )
        
        return workflow.compile()

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        import asyncio

        # Create subgraphs for selected analysts
        analyst_graphs = {}
        
        if "market" in selected_analysts:
            analyst_graphs["market"] = self._create_analyst_subgraph(
                "market",
                create_market_analyst(self.quick_thinking_llm),
                self.tool_nodes["market"],
                self.conditional_logic.should_continue_market
            )

        if "social" in selected_analysts:
            analyst_graphs["social"] = self._create_analyst_subgraph(
                "social",
                create_social_media_analyst(self.quick_thinking_llm),
                self.tool_nodes["social"],
                self.conditional_logic.should_continue_social
            )

        if "news" in selected_analysts:
            analyst_graphs["news"] = self._create_analyst_subgraph(
                "news",
                create_news_analyst(self.quick_thinking_llm),
                self.tool_nodes["news"],
                self.conditional_logic.should_continue_news
            )

        if "fundamentals" in selected_analysts:
            analyst_graphs["fundamentals"] = self._create_analyst_subgraph(
                "fundamentals",
                create_fundamentals_analyst(self.quick_thinking_llm),
                self.tool_nodes["fundamentals"],
                self.conditional_logic.should_continue_fundamentals
            )

        # Define the parallel execution node
        async def run_analysts_parallel(state):
            # Create a clean state for subgraphs (empty messages)
            # We want to keep the input fields (ticker, date) but clear history
            # Note: We must ensure we don't pass the full 'messages' history to analysts
            # as they should start fresh.
            sub_state = state.copy()
            sub_state["messages"] = []
            
            # Prepare tasks
            tasks = []
            keys = []
            
            # Order matters for matching results to keys, though we use a dict below
            active_analysts = [k for k in ["market", "social", "news", "fundamentals"] if k in analyst_graphs]
            
            for key in active_analysts:
                tasks.append(analyst_graphs[key].ainvoke(sub_state))
                keys.append(key)
            
            # Run in parallel
            results = await asyncio.gather(*tasks)
            
            # Merge results
            updates = {}
            key_map = {
                "market": "market",
                "social": "sentiment",
                "news": "news",
                "fundamentals": "fundamentals"
            }
            
            for key, res in zip(keys, results):
                prefix = key_map.get(key, key)
                report_key = f"{prefix}_report"
                data_key = f"{prefix}_data"
                
                if report_key in res:
                    updates[report_key] = res[report_key]
                if data_key in res:
                    updates[data_key] = res[data_key]
            
            # We also need to clear the messages from the main state effectively
            # or rather, we just return the updates. The main state's messages 
            # (which might be just the user input) are preserved unless we overwrite them.
            # We probably want to clear the messages to prepare for the next stage (Bull Researcher).
            # So we return a message removal operation or just set messages to empty?
            # The 'create_msg_delete' used to do this.
            # Let's return a placeholder message to keep the history clean.
            from langchain_core.messages import HumanMessage, RemoveMessage
            removal_operations = [RemoveMessage(id=m.id) for m in state["messages"]]
            placeholder = HumanMessage(content="Analysts have completed their reports.")
            
            updates["messages"] = removal_operations + [placeholder]
            
            return updates

        # Create researcher and manager nodes
        critic_node = create_critic_agent(self.deep_thinking_llm)
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add the parallel node
        workflow.add_node("Parallel Analysts", run_analysts_parallel)
        workflow.add_node("Critic", critic_node)

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges
        # Start -> Parallel Analysts
        workflow.add_edge(START, "Parallel Analysts")
        
        # Parallel Analysts -> Critic -> Bull Researcher
        workflow.add_edge("Parallel Analysts", "Critic")
        workflow.add_edge("Critic", "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
