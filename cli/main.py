from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.latex_utils import escape_latex as _latex_escape
import warnings

# Suppress yfinance FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        if any(
            self.report_sections[section]
            for section in [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections(ticker: Optional[str] = None, quick: bool = False):
    """Get all user selections before starting the analysis display."""
    if quick:
        # Quick mode: use defaults
        selected_ticker = ticker or get_ticker()
        analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")
        lookback_days = DEFAULT_CONFIG.get("analysis_window_days", 7)
        selected_analysts = [a for a in AnalystType]
        selected_research_depth = "comprehensive"
        selected_llm_provider = "openai"
        backend_url = None
        selected_shallow_thinker = "gpt-4o-mini"
        selected_deep_thinker = "gpt-4o"
        
        console.print(f"[green]Quick Start Mode:[/green] Analyzing {selected_ticker} with default settings.")
        
        return {
            "ticker": selected_ticker,
            "analysis_date": analysis_date,
            "analysts": selected_analysts,
            "research_depth": selected_research_depth,
            "llm_provider": selected_llm_provider,
            "backend_url": backend_url,
            "shallow_thinker": selected_shallow_thinker,
            "deep_thinker": selected_deep_thinker,
            "lookback_days": lookback_days,
        }

    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Lookback window
    default_window = str(DEFAULT_CONFIG.get("analysis_window_days", 7))
    console.print(
        create_question_box(
            "Step 3: Lookback Window",
            "Enter how many days of history to analyze (e.g., 7 for a week, 30 for a month)",
            f"{default_window} days",
        )
    )
    lookback_days = get_analysis_window()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: Provider backend
    console.print(
        create_question_box(
            "Step 6: LLM Provider Backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 7: Thinking agents
    console.print(
        create_question_box(
            "Step 7: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "lookback_days": lookback_days,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def get_analysis_window():
    """Prompt for number of days to analyze."""
    default_value = str(DEFAULT_CONFIG.get("analysis_window_days", 7))
    while True:
        response = typer.prompt("", default=default_value)
        try:
            lookback_days = int(response)
            if lookback_days <= 0:
                console.print("[red]Error: Lookback window must be a positive number of days[/red]")
                continue
            if lookback_days > 365:
                console.print("[yellow]Warning: Large lookback windows may slow the analysis; continuing anyway[/yellow]")
            return lookback_days
        except ValueError:
            console.print("[red]Error: Please enter a whole number of days[/red]")


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Portfolio Manager Decision
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


from collections import Counter


def _format_markdown_table(rows):
    table = ["| Metric | Value |", "| --- | --- |"]
    for metric, value in rows:
        rendered = "-" if value is None else str(value)
        rendered = rendered.replace("\n", "<br>")
        table.append(f"| {metric} | {rendered} |")
    return table


def _get_reasoning_entries(message_buffer, limit=10):
    return [entry for entry in message_buffer.messages if entry[1] in {"Reasoning", "Analysis"}][-limit:]


def _get_tool_log_entries(message_buffer, limit=10):
    tool_calls = list(message_buffer.tool_calls)
    return tool_calls[-limit:]


def _build_markdown_report(final_state, selections, decision, message_buffer, config):
    lines = [
        f"# TradingAgents Report for {selections['ticker']} ({selections['analysis_date']})",
        "",
    ]

    key_rows = [
        ("Ticker", selections["ticker"]),
        ("Trade Date", selections["analysis_date"]),
        ("Final Decision", decision or "N/A"),
        ("Quick-Thinking Model", selections["shallow_thinker"]),
        ("Deep-Thinking Model", selections["deep_thinker"]),
        ("Research Depth", selections.get("research_depth", "-")),
        ("Lookback Window", f"{selections['lookback_days']} days"),
        ("Analysts Engaged", ", ".join(a.value for a in selections["analysts"])),
    ]
    lines.append("## Key Metrics")
    lines.extend(_format_markdown_table(key_rows))
    lines.append("")

    if config.get("data_vendors"):
        lines.append("### Data Vendors")
        for category, vendor in config["data_vendors"].items():
            lines.append(f"- **{category}** → {vendor}")
        if config.get("tool_vendors"):
            lines.append("- **Tool Overrides**")
            for tool, vendor in config["tool_vendors"].items():
                lines.append(f"  - `{tool}` → {vendor}")
        lines.append("")

    def add_section(title, content):
        if content:
            lines.append(f"## {title}")
            lines.append(content.strip())
            lines.append("")

    add_section("Market Analysis", final_state.get("market_report"))
    add_section("Social Sentiment", final_state.get("sentiment_report"))
    add_section("News Analysis", final_state.get("news_report"))
    add_section("Fundamentals", final_state.get("fundamentals_report"))

    debate_state = final_state.get("investment_debate_state") or {}
    add_section("Bull Researcher", debate_state.get("bull_history"))
    add_section("Bear Researcher", debate_state.get("bear_history"))
    add_section("Research Manager Decision", debate_state.get("judge_decision"))

    add_section("Trading Plan", final_state.get("trader_investment_plan"))

    risk_state = final_state.get("risk_debate_state") or {}
    add_section("Aggressive (Risky) Analyst", risk_state.get("risky_history"))
    add_section("Neutral Analyst", risk_state.get("neutral_history"))
    add_section("Conservative (Safe) Analyst", risk_state.get("safe_history"))
    add_section("Risk Committee Outcome", risk_state.get("judge_decision"))

    add_section("Final Portfolio Decision", final_state.get("final_trade_decision"))

    lines.append("## Portfolio Impact")
    impact_rows = [
        ("Action", decision or "Pending"),
        ("Portfolio Guidance", risk_state.get("judge_decision") or "See decision text"),
        ("Execution Notes", final_state.get("final_trade_decision", "N/A")),
    ]
    lines.extend(_format_markdown_table(impact_rows))
    lines.append("")

    tool_counter = Counter(name for _, name, _ in message_buffer.tool_calls)
    if tool_counter:
        lines.append("## Tool Usage Summary")
        tool_rows = [(tool, str(count)) for tool, count in tool_counter.most_common()]
        lines.extend(_format_markdown_table(tool_rows))
        lines.append("")

    reasoning_entries = _get_reasoning_entries(message_buffer)
    tool_entries = _get_tool_log_entries(message_buffer)

    lines.append("## Trace Appendix")
    if reasoning_entries:
        lines.append("### Reasoning Timeline (most recent)")
        for timestamp, _, content in reasoning_entries:
            lines.append(f"- **{timestamp}** — {content.strip()}")
        lines.append("")
    if tool_entries:
        lines.append("### Tool Calls")
        for timestamp, tool_name, args in tool_entries:
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            lines.append(f"- **{timestamp}** — `{tool_name}` ({arg_str})")
        lines.append("")

    return "\n".join(lines).strip() + "\n"





def _latex_table(rows):
    lines = [r"\begin{tabular}{p{0.3\linewidth} p{0.6\linewidth}}", r"\textbf{Metric} & \textbf{Value} \\ \hline"]
    for metric, value in rows:
        rendered = "-" if value is None else str(value)
        rendered = _latex_escape(rendered).replace("\n", " \\ ")
        lines.append(f"{_latex_escape(metric)} & {rendered} \\ ")
    lines.append(r"\end{tabular}")
    return lines


def _build_latex_report(final_state, selections, decision, message_buffer, config):
    def latex_section(title, content):
        if not content:
            return []
        body = _latex_escape(content)
        return [f"\\section*{{{_latex_escape(title)}}}", body, ""]

    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{parskip}",
        r"\begin{document}",
        fr"\section*{{TradingAgents Report for {_latex_escape(selections['ticker'])} ({_latex_escape(selections['analysis_date'])})}}",
        r"\subsection*{Configuration}",
        r"\begin{itemize}",
        fr"\item LLM Provider: {_latex_escape(selections['llm_provider'].title())}",
        fr"\item Quick-Thinking Model: {_latex_escape(selections['shallow_thinker'])}",
        fr"\item Deep-Thinking Model: {_latex_escape(selections['deep_thinker'])}",
        fr"\item Lookback Window: {selections['lookback_days']} days",
        r"\end{itemize}",
        "",
    ]

    key_rows = [
        ("Ticker", selections["ticker"]),
        ("Trade Date", selections["analysis_date"]),
        ("Final Decision", decision or "N/A"),
        ("Quick-Thinking Model", selections["shallow_thinker"]),
        ("Deep-Thinking Model", selections["deep_thinker"]),
        ("Research Depth", selections.get("research_depth", "-")),
        ("Lookback Window", f"{selections['lookback_days']} days"),
        ("Analysts Engaged", ", ".join(a.value for a in selections["analysts"])),
    ]
    lines.append(r"\subsection*{Key Metrics}")
    lines.extend(_latex_table(key_rows))
    lines.append("")

    if config.get("data_vendors"):
        lines.append(r"\subsection*{Data Vendors}")
        lines.append(r"\begin{itemize}")
        for category, vendor in config["data_vendors"].items():
            lines.append(fr"\item {_latex_escape(category)} \rightarrow {_latex_escape(vendor)}")
        if config.get("tool_vendors"):
            lines.append(r"\item Overrides:")
            lines.append(r"\begin{itemize}")
            for tool, vendor in config["tool_vendors"].items():
                lines.append(fr"\item {_latex_escape(tool)} \rightarrow {_latex_escape(vendor)}")
            lines.append(r"\end{itemize}")
        lines.append(r"\end{itemize}")
        lines.append("")

    sections = [
        ("Market Analysis", final_state.get("market_report")),
        ("Social Sentiment", final_state.get("sentiment_report")),
        ("News Analysis", final_state.get("news_report")),
        ("Fundamentals", final_state.get("fundamentals_report")),
    ]

    debate_state = final_state.get("investment_debate_state") or {}
    sections.extend(
        [
            ("Bull Researcher", debate_state.get("bull_history")),
            ("Bear Researcher", debate_state.get("bear_history")),
            ("Research Manager Decision", debate_state.get("judge_decision")),
        ]
    )

    sections.append(("Trading Plan", final_state.get("trader_investment_plan")))

    risk_state = final_state.get("risk_debate_state") or {}
    sections.extend(
        [
            ("Aggressive (Risky) Analyst", risk_state.get("risky_history")),
            ("Neutral Analyst", risk_state.get("neutral_history")),
            ("Conservative (Safe) Analyst", risk_state.get("safe_history")),
            ("Risk Committee Outcome", risk_state.get("judge_decision")),
        ]
    )

    sections.append(("Final Portfolio Decision", final_state.get("final_trade_decision")))

    for title, content in sections:
        lines.extend(latex_section(title, content))

    risk_state = final_state.get("risk_debate_state") or {}
    lines.append(r"\subsection*{Portfolio Impact}")
    impact_rows = [
        ("Action", decision or "Pending"),
        ("Portfolio Guidance", risk_state.get("judge_decision") or "See decision text"),
        ("Execution Notes", final_state.get("final_trade_decision", "N/A")),
    ]
    lines.extend(_latex_table(impact_rows))
    lines.append("")

    tool_counter = Counter(name for _, name, _ in message_buffer.tool_calls)
    if tool_counter:
        lines.append(r"\subsection*{Tool Usage Summary}")
        tool_rows = [(tool, str(count)) for tool, count in tool_counter.most_common()]
        lines.extend(_latex_table(tool_rows))
        lines.append("")

    reasoning_entries = _get_reasoning_entries(message_buffer)
    tool_entries = _get_tool_log_entries(message_buffer)

    if reasoning_entries:
        lines.append(r"\subsection*{Reasoning Timeline}")
        lines.append(r"\begin{itemize}")
        for timestamp, _, content in reasoning_entries:
            lines.append(fr"\item \textbf{{{_latex_escape(timestamp)}}} -- {_latex_escape(content)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    if tool_entries:
        lines.append(r"\subsection*{Tool Call Log}")
        lines.append(r"\begin{itemize}")
        for timestamp, tool_name, args in tool_entries:
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            lines.append(fr"\item \textbf{{{_latex_escape(timestamp)}}} -- \texttt{{{_latex_escape(tool_name)}}} ({_latex_escape(arg_str)})")
        lines.append(r"\end{itemize}")
        lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def save_structured_reports(final_state, selections, report_dir, message_buffer, decision, config):
    """Persist the investment memo as a single LaTeX file."""
    import logging
    
    # Configure logging to file
    log_file = report_dir / "report_generation.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting report generation sequence...")
    
    # Validate final_state
    required_keys = ["market_report", "fundamentals_report", "final_trade_decision"]
    missing_keys = [k for k in required_keys if not final_state.get(k)]
    if missing_keys:
        logger.warning(f"Missing keys in final_state: {missing_keys}")
        console.print(f"[yellow]Warning: Missing data for report: {missing_keys}[/yellow]")

    from cli.report_generator import build_latex_report, build_markdown_report
    from cli.latex_utils import save_latex_debug

    try:
        latex_report = build_latex_report(final_state, selections, decision, report_dir)
        (report_dir / "final_report.tex").write_text(latex_report)
        save_latex_debug(latex_report, report_dir / "debug_final_report.tex")
        console.print(f"[green]Saved LaTeX report to {report_dir / 'final_report.tex'}[/green]")
    except Exception as e:
        logger.error(f"Failed to generate LaTeX report: {e}", exc_info=True)
        console.print(f"[red]Failed to generate LaTeX report. See {log_file} for details.[/red]")

    try:
        markdown_report = build_markdown_report(final_state, selections, decision)
        (report_dir / "final_report.md").write_text(markdown_report)
        console.print(f"[green]Saved Markdown report to {report_dir / 'final_report.md'}[/green]")
    except Exception as e:
        logger.error(f"Failed to generate Markdown report: {e}", exc_info=True)
        console.print(f"[red]Failed to generate Markdown report. See {log_file} for details.[/red]")


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

import asyncio

async def run_analysis_async(selections: dict):
    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    config["analysis_window_days"] = selections["lookback_days"]

    # Initialize the graph
    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        # Reset agent statuses
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        # Reset report sections
        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        # Initialize state and get graph args
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        # Stream the analysis
        trace = []
        async for chunk in graph.graph.astream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # Add message to buffer
                message_buffer.add_message(msg_type, content)                

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    # Set next analyst to in_progress
                    if "fundamentals" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Fundamentals Analyst", "in_progress"
                        )

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    # Set all research team members to in_progress
                    update_research_team_status("in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            # Update research report with bull's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            # Update research report with bear's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        # Keep all research team members in progress until final decision
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        # Update research report with final decision
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        # Mark all research team members as completed
                        update_research_team_status("completed")
                        # Set first risk analyst to in_progress
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final decision
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final decision only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                # Update the display
                update_display(layout)

            trace.append(chunk)

        # Get final state and decision
        if not trace:
            console.print("[red]Error: No execution trace generated. The graph may have failed to start.[/red]")
            return

        final_state = trace[-1]
        
        decision = "Hold" # Default decision
        if "final_trade_decision" in final_state and final_state["final_trade_decision"]:
            try:
                decision = graph.process_signal(final_state["final_trade_decision"])
            except Exception as e:
                console.print(f"[yellow]Warning: Could not process signal: {e}. Defaulting to 'Hold'.[/yellow]")
        else:
            console.print("[yellow]Warning: No final trade decision found in state. Defaulting to 'Hold'.[/yellow]")

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        # Display the complete final report
        display_complete_report(final_state)
        save_structured_reports(final_state, selections, report_dir, message_buffer, decision, config)

        update_display(layout)


@app.command()
def analyze(
    ticker: Optional[str] = typer.Option(None, help="Ticker symbol to analyze"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Skip questionnaire and use defaults"),
):
    # First get all user selections synchronously (avoids nested event loop issues)
    selections = get_user_selections(ticker=ticker, quick=quick)
    
    # Then run the async analysis
    asyncio.run(run_analysis_async(selections))


if __name__ == "__main__":
    app()
