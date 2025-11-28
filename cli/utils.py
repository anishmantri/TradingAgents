import questionary
from typing import List, Optional, Tuple, Dict

from cli.models import AnalystType

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5.1 Mini - Lightweight frontier reasoning", "gpt-5.1-mini"),
            ("GPT-5.1 - Flagship frontier model", "gpt-5.1"),
            ("GPT-5 Mini - Efficient 5-series option", "gpt-5-mini"),
            ("GPT-5 - General-purpose 5-series model", "gpt-5"),
            ("GPT-4.1 Mini - Balanced lightweight model", "gpt-4.1-mini"),
            ("GPT-4.1 - Proven general-purpose model", "gpt-4.1"),
            ("GPT-4o Mini - Fast multi-modal option", "gpt-4o-mini"),
            ("GPT-4o - Standard multi-modal model", "gpt-4o"),
        ],
        "anthropic": [
            ("Claude 3.5 Haiku - Fast inference tier", "claude-3-5-haiku-latest"),
            ("Claude 3.5 Sonnet - Balanced reasoning", "claude-3-5-sonnet-latest"),
            ("Claude 3.7 Sonnet - Hybrid reasoning + agentic workflows", "claude-3-7-sonnet-latest"),
            ("Claude Sonnet 4 - High performance and excellent reasoning", "claude-sonnet-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash Lite - Lowest-latency option", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash - Fast multi-modal", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash Preview - Adaptive lightweight reasoning", "gemini-2.5-flash-preview-05-20"),
        ],
        "openrouter": [
            ("OpenAI GPT-5.1 - Flagship general model", "openai/gpt-5.1"),
            ("OpenAI GPT-5 Mini - Lightweight 5-series option", "openai/gpt-5-mini"),
            ("Gemini 2.0 Flash Experimental (free)", "google/gemini-2.0-flash-exp:free"),
            ("Grok 4.1 Fast (free) - xAI agentic model", "x-ai/grok-4.1-fast:free"),
            ("Kimi K2 Thinking - MoonshotAI reasoning tuned", "moonshotai/kimi-k2-thinking"),
            ("Meta Llama 4 Scout (free) - Efficient Llama 4 preview", "meta-llama/llama-4-scout:free"),
        ],
        "ollama": [
            ("llama3.1 local", "llama3.1"),
            ("llama3.2 local", "llama3.2"),
        ],
    }

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    DEEP_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5.1 Pro - Frontier long-form reasoning", "gpt-5.1-pro"),
            ("GPT-5 Pro - Heavyweight reasoning", "gpt-5-pro"),
            ("GPT-5.1 - Flagship generalist", "gpt-5.1"),
            ("GPT-5 - Advanced 5-series generalist", "gpt-5"),
            ("o4-mini - Latest compact reasoning model", "o4-mini"),
            ("o3-mini - Advanced lightweight reasoning", "o3-mini"),
            ("o1 - Premier reasoning and problem-solving model", "o1"),
            ("GPT-4.1 - Full-featured generalist", "gpt-4.1"),
        ],
        "anthropic": [
            ("Claude 3.7 Sonnet (latest)", "claude-3-7-sonnet-latest"),
            ("Claude 3.5 Sonnet (latest)", "claude-3-5-sonnet-latest"),
            ("Claude 3.5 Haiku - cost effective reasoning", "claude-3-5-haiku-latest"),
            ("Claude Opus 4 - Maximum capability", "claude-opus-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash - Balanced multi-modal reasoning", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash Preview - Adaptive reasoning", "gemini-2.5-flash-preview-05-20"),
            ("Gemini 2.5 Pro Preview", "gemini-2.5-pro-preview-06-05"),
        ],
        "openrouter": [
            ("OpenAI GPT-5 Pro - Heavyweight reasoning", "openai/gpt-5-pro"),
            ("OpenAI GPT-5.1 - Frontier general reasoning", "openai/gpt-5.1"),
            ("DeepSeek R1 Distill Llama 70B (free) - Deep reasoning", "deepseek/deepseek-r1-distill-llama-70b:free"),
            ("Kimi K2 Thinking - MoonshotAI frontier reasoning", "moonshotai/kimi-k2-thinking"),
            ("Prime Intellect INTELLECT-3 - 106B MoE frontier", "prime-intellect/intellect-3"),
            ("Tongyi DeepResearch 30B A3B (free) - Alibaba research agent", "alibaba/tongyi-deepresearch-30b-a3b:free"),
        ],
        "ollama": [
            ("llama3.1 local", "llama3.1"),
            ("qwen3", "qwen3"),
        ]
    }
    
    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str]:
    """Select the LLM provider endpoint using interactive selection."""
    # Define provider api options with their corresponding endpoints
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Anthropic", "https://api.anthropic.com/"),
        ("Google", "https://generativelanguage.googleapis.com/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),        
    ]
    
    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]no OpenAI backend selected. Exiting...[/red]")
        exit(1)
    
    display_name, url = choice
    print(f"You selected: {display_name}\tURL: {url}")
    
    return display_name, url
