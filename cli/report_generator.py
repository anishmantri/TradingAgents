"""Generate a structured LaTeX report for the TradingAgents CLI.

The builder assembles a hedge-fund style memo with data pulled from yfinance
and the agent outputs. Graphs are rendered with PGFPlots inside LaTeX so the
output is a single self-contained ``.tex`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import re
import time
import logging
import concurrent.futures

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# --------------------------- Data Containers ---------------------------


@dataclass
class PriceSnapshot:
    dates: List[str]
    closes: List[float]
    one_year_return: Optional[float]
    three_month_return: Optional[float]
    volatility: Optional[float]
    beta: Optional[float]
    rsi: Optional[List[float]]
    macd: Optional[List[float]]
    macd_signal: Optional[List[float]]


@dataclass
class FinancialSnapshot:
    revenue_series: List[Tuple[str, float]]
    ebitda_series: List[Tuple[str, float]]
    margin_series: List[Tuple[str, float]]
    fcf_series: List[Tuple[str, float]]
    net_debt: Optional[float]
    shares_outstanding: Optional[float]
    
    # Snapshot values for the table
    latest_revenue: Optional[float]
    latest_ebitda: Optional[float]
    latest_net_income: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    
    # Balance Sheet
    total_cash: Optional[float]
    total_debt: Optional[float]
    total_assets: Optional[float]
    total_liabilities: Optional[float]
    equity: Optional[float]


@dataclass
class ValuationSnapshot:
    price: Optional[float]
    market_cap: Optional[float]
    enterprise_value: Optional[float]
    pe: Optional[float]
    ev_ebitda: Optional[float]
    fcf_yield: Optional[float]
    growth: Optional[float]
    dividend_yield: Optional[float]


@dataclass
class ScenarioCase:
    name: str
    price: float
    return_pct: float
    probability: float
    reasoning: str


def _clean_agent_text(text: str) -> str:
    """Cleans up agent output to remove role labels and artifacts."""
    if not text:
        return ""
    
    # Remove role labels like "Risky Analyst:", "Market Analyst:", etc.
    # Added \s* to handle leading whitespace
    text = re.sub(r'^\s*(Market|Social|News|Fundamentals|Risky|Neutral|Safe) Analyst:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*(Bull|Bear) Researcher:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*Research Manager:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*Portfolio Manager:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove "Sub-agent" mentions
    text = re.sub(r'Sub-agent', 'Analyst', text, flags=re.IGNORECASE)
    
    # Remove "Price()" artifact if present
    text = re.sub(r'Price\(\)', 'Price', text, flags=re.IGNORECASE)
    
    # Remove "newline" artifacts
    text = re.sub(r'\bnewline\b', ' ', text, flags=re.IGNORECASE)
    
    # Remove "Here is the report:" type intros
    text = re.sub(r'^\s*Here is the (report|analysis|summary).*?:', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text.strip()


# --------------------------- Helpers ---------------------------


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return None
        return num
    except Exception:
        return None


def _clean_series(series: pd.Series, limit: int = 12) -> List[Tuple[str, float]]:
    if series is None or series.empty:
        return []
    ordered = series.dropna()
    if ordered.empty:
        return []
    ordered = ordered.sort_index()
    recent = ordered.tail(limit)
    cleaned: List[Tuple[str, float]] = []
    for idx, val in recent.items():
        label = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        if pd.isna(val):
            continue
        cleaned.append((label, float(val)))
    return cleaned


def _compute_beta(asset_returns: pd.Series, bench_returns: pd.Series) -> Optional[float]:
    if asset_returns.empty or bench_returns.empty:
        return None
    joined = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if joined.empty:
        return None
    cov = joined.cov().iloc[0, 1]
    var_bench = joined.var().iloc[1]
    if var_bench == 0:
        return None
    return float(cov / var_bench)


def _scale_series(data: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], str]:
    """Scales a series of values to Billions or Millions if appropriate."""
    if not data:
        return [], ""
    
    max_val = max(abs(v) for _, v in data)
    
    if max_val >= 1e9:
        return [(l, v / 1e9) for l, v in data], "($B)"
    elif max_val >= 1e6:
        return [(l, v / 1e6) for l, v in data], "($M)"
    return data, ""


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"{val*100:.1f}\\%"


def _fmt_curr(val: Optional[float], short: bool = False) -> str:
    if val is None:
        return "-"
    if abs(val) >= 1e12:
        return f"\\${val/1e12:,.2f}T"
    if abs(val) >= 1e9:
        return f"\\${val/1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"\\${val/1e6:,.1f}M"
    return f"\\${val:,.0f}"


def _latex_escape(text: str) -> str:
    if not text:
        return ""
    
    # First, handle the specific case of "newline" text artifacts
    # If the text contains literal "newline" surrounded by spaces or at start/end, remove it
    text = re.sub(r'\s*newline\s*', ' ', str(text), flags=re.IGNORECASE)
    
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
        "<": r"\\textless{}",
        ">": r"\\textgreater{}",
    }
    
    safe_text = ""
    for char in text:
        if char in replacements:
            safe_text += replacements[char]
        else:
            safe_text += char
            
    return safe_text


def _latex_format_text(text: str) -> str:
    """Parses Markdown-like text and converts it to LaTeX."""
    if not text:
        return ""
    
    # Normalize newlines
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove "newline" artifacts
    text = re.sub(r'\bnewline\b', ' ', text, flags=re.IGNORECASE)
    
    lines = text.split("\n")
    formatted_lines = []
    in_list = False
    in_code_block = False
    in_table = False
    table_rows = []
    
    for line in lines:
        stripped = line.strip()
        
        # Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                formatted_lines.append(r"\end{verbatim}")
                in_code_block = False
            else:
                if in_list:
                    formatted_lines.append(r"\end{itemize}")
                    in_list = False
                formatted_lines.append(r"\begin{verbatim}")
                in_code_block = True
            continue
            
        if in_code_block:
            formatted_lines.append(line)
            continue

        # Tables (basic support)
        if "|" in line and (line.strip().startswith("|") or line.strip().endswith("|")):
            if not in_table:
                in_table = True
                table_rows = []
            
            # Skip separator lines like |---|---|
            if set(line.strip()) <= {"|", "-", " ", ":"}:
                continue
                
            # Parse row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        elif in_table:
            # End of table
            in_table = False
            if table_rows:
                num_cols = len(table_rows[0])
                col_spec = "l" * num_cols
                formatted_lines.append(r"\begin{tabular}{" + col_spec + "}")
                formatted_lines.append(r"\toprule")
                
                # Header
                header_cells = [_process_formatting(c) for c in table_rows[0]]
                formatted_lines.append(" & ".join(rf"\textbf{{{c}}}" for c in header_cells) + r" \\")
                formatted_lines.append(r"\midrule")
                
                # Body
                for row in table_rows[1:]:
                    # Ensure row has same number of columns
                    if len(row) != num_cols:
                        continue
                    row_cells = [_process_formatting(c) for c in row]
                    formatted_lines.append(" & ".join(row_cells) + r" \\")
                    
                formatted_lines.append(r"\bottomrule")
                formatted_lines.append(r"\end{tabular}")
                formatted_lines.append(r"\par\vspace{0.5em}")
            table_rows = []

        if not stripped:
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            formatted_lines.append(r"\par\vspace{0.5em}") 
            continue
            
        # Headers
        if stripped.startswith("###"):
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            content = stripped.lstrip("#").strip()
            formatted_lines.append(rf"\paragraph*{{{_latex_escape(content)}}}")
            continue
        elif stripped.startswith("##"):
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            content = stripped.lstrip("#").strip()
            formatted_lines.append(rf"\subsubsection*{{{_latex_escape(content)}}}")
            continue
            
        # List items
        is_list_item = stripped.startswith(("- ", "* ", "• "))
        if is_list_item:
            content = stripped[1:].strip()
            if not in_list:
                formatted_lines.append(r"\begin{itemize}")
                in_list = True
            
            content = _process_formatting(content)
            formatted_lines.append(rf"\item {content}")
        else:
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            
            content = _process_formatting(stripped)
            formatted_lines.append(content + r" \\")
            
    if in_list:
        formatted_lines.append(r"\end{itemize}")
    if in_code_block:
        formatted_lines.append(r"\end{verbatim}")
    if in_table and table_rows:
        # Flush remaining table if file ended
        num_cols = len(table_rows[0])
        col_spec = "l" * num_cols
        formatted_lines.append(r"\begin{tabular}{" + col_spec + "}")
        formatted_lines.append(r"\toprule")
        header_cells = [_process_formatting(c) for c in table_rows[0]]
        formatted_lines.append(" & ".join(rf"\textbf{{{c}}}" for c in header_cells) + r" \\")
        formatted_lines.append(r"\midrule")
        for row in table_rows[1:]:
             if len(row) == num_cols:
                row_cells = [_process_formatting(c) for c in row]
                formatted_lines.append(" & ".join(row_cells) + r" \\")
        formatted_lines.append(r"\bottomrule")
        formatted_lines.append(r"\end{tabular}")
        
    return "\n".join(formatted_lines)


def _process_formatting(text: str) -> str:
    """Helper to handle **bold** and *italic* text."""
    # Escape special characters first
    escaped = _latex_escape(text)
    
    # Bold: **text** -> \textbf{text}
    escaped = re.sub(r'\*\*(.*?)\*\*', r"\\textbf{\1}", escaped)
    
    # Italic: *text* -> \textit{text}
    # Use negative lookbehind/lookahead to avoid matching ** as *
    escaped = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r"\\textit{\1}", escaped)
    
    return escaped


def _pgf_time_series(
    coords: List[Tuple[str, float]], 
    title: str, 
    ylabel: str, 
    date_labels: bool = False,
    extra_coords: Optional[List[Tuple[str, float]]] = None,
    legend_labels: Optional[List[str]] = None
) -> str:
    if not coords:
        return "\\textit{Insufficient data for chart.}"
    
    xs = list(range(len(coords)))
    step = max(1, len(coords) // 8)
    xticks = " ,".join(str(i) for i in xs[::step])
    xticklabels = " ,".join(label for label, _ in coords[::step])
    
    points = " ".join(f"({i},{y:.2f})" for i, (_, y) in zip(xs, coords))
    
    latex = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "    width=0.95\\linewidth,\n"
        "    height=6cm,\n"
        "    grid=both,\n"
        "    grid style={dashed,gray!30},\n"
        f"    title={{{{{title}}}}},\n"
        f"    ylabel={{{{{ylabel}}}}},\n"
        f"    xtick={{{{{xticks}}}}},\n"
        f"    xticklabels={{{{{xticklabels}}}}},\n"
        "    xticklabel style={rotate=45, anchor=east, font=\\footnotesize},\n"
        "    ylabel style={font=\\footnotesize},\n"
        "    title style={font=\\bfseries},\n"
        "    legend style={at={(0.02,0.98)},anchor=north west, font=\\footnotesize},\n"
        "    scaled y ticks=false,\n"
        "    y tick label style={/pgf/number format/fixed}\n"
        "]\n"
        f"\\addplot+[mark=*,mark size=1.5pt,thick,color=blue!70!black] coordinates {{{points}}};\n"
    )
    
    if legend_labels:
        latex += f"\\addlegendentry{{{legend_labels[0]}}}\n"

    if extra_coords:
        points2 = " ".join(f"({i},{y:.2f})" for i, (_, y) in zip(xs, extra_coords))
        latex += f"\\addplot+[mark=none,thick,color=red!70!black] coordinates {{{points2}}};\n"
        if legend_labels and len(legend_labels) > 1:
            latex += f"\\addlegendentry{{{legend_labels[1]}}}\n"

    latex += "\\end{axis}\n\\end{tikzpicture}"
    return latex


def _pgf_bar_chart(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> str:
    if not labels or not values:
        return "\\textit{Insufficient data for chart.}"
    
    coords = " ".join(f"({i},{v:.2f})" for i, v in enumerate(values))
    xticks = " ,".join(str(i) for i in range(len(labels)))
    xticklabels = " ,".join(labels)
    
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "    ybar,\n"
        "    bar width=15pt,\n"
        "    width=0.95\\linewidth,\n"
        "    height=6cm,\n"
        "    grid=major,\n"
        "    grid style={dashed,gray!30},\n"
        "    enlarge x limits=0.15,\n"
        "    nodes near coords,\n"
        "    nodes near coords style={font=\\tiny, color=black, /pgf/number format/fixed, /pgf/number format/precision=1},\n"
        f"    title={{{{{title}}}}},\n"
        f"    ylabel={{{{{ylabel}}}}},\n"
        f"    xtick={{{{{xticks}}}}},\n"
        f"    xticklabels={{{{{xticklabels}}}}},\n"
        "    xticklabel style={rotate=45, anchor=east, font=\\footnotesize},\n"
        "    title style={font=\\bfseries},\n"
        "    scaled y ticks=false,\n"
        "    y tick label style={/pgf/number format/fixed}\n"
        "]\n"
        f"\\addplot+[fill=blue!60!white, draw=blue!80!black] coordinates {{{coords}}};\n"
        "\\end{axis}\n\\end{tikzpicture}"
    )


def load_price_snapshot(ticker: str, analysis_date: str, lookback_days: int) -> PriceSnapshot:
    start_time = time.time()
    logger.info(f"Loading price snapshot for {ticker}...")
    try:
        end_dt = pd.to_datetime(analysis_date)
        start_dt = end_dt - timedelta(days=max(lookback_days, 365))
        price_df = yf.download(ticker, start=start_dt, end=end_dt + timedelta(days=1), progress=False)
        if price_df.empty:
            return PriceSnapshot([], [], None, None, None, None, None, None, None)

        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)

        price_df = price_df.sort_index()
        close_candidates = ["Adj Close", "Close", "adjclose", "adj_close", "close"]
        close_col = next((c for c in close_candidates if c in price_df.columns), None)
        if not close_col:
            return PriceSnapshot([], [], None, None, None, None, None, None, None)

        price_df = price_df.dropna(subset=[close_col])
        closes = price_df[close_col]
        dates = [idx.strftime("%Y-%m-%d") for idx in closes.index]

        ret_1y = None
        ret_3m = None
        if len(closes) > 60:
            ret_3m = float(closes.iloc[-1] / closes.iloc[-63] - 1)
        if len(closes) > 250:
            ret_1y = float(closes.iloc[-1] / closes.iloc[-252] - 1)

        returns = closes.pct_change().dropna()
        vol = float(returns.std() * np.sqrt(252)) if not returns.empty else None

        spy_df = yf.download("SPY", start=start_dt, end=end_dt + timedelta(days=1), progress=False)
        beta = None
        if not spy_df.empty:
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            spy_close_col = next((c for c in close_candidates if c in spy_df.columns), None)
            if spy_close_col:
                spy_ret = spy_df[spy_close_col].pct_change().dropna()
                beta = _compute_beta(returns, spy_ret)

        # Technical Indicators
        # RSI (14)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = rsi_series.fillna(50).tolist() # Fill NaN with 50 (neutral)

        # MACD (12, 26, 9)
        exp1 = closes.ewm(span=12, adjust=False).mean()
        exp2 = closes.ewm(span=26, adjust=False).mean()
        macd_series = exp1 - exp2
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        
        macd = macd_series.fillna(0).tolist()
        macd_signal = signal_series.fillna(0).tolist()

        logger.info(f"Loaded price snapshot for {ticker} in {time.time() - start_time:.2f}s")
        return PriceSnapshot(
            dates=dates, 
            closes=list(closes.values), 
            one_year_return=ret_1y, 
            three_month_return=ret_3m, 
            volatility=vol, 
            beta=beta,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal
        )
    except Exception as e:
        logger.error(f"Error loading price snapshot for {ticker}: {e}")
        return PriceSnapshot([], [], None, None, None, None, None, None, None)


def load_financial_snapshot(ticker: str) -> FinancialSnapshot:
    start_time = time.time()
    logger.info(f"Loading financial snapshot for {ticker}...")
    try:
        tkr = yf.Ticker(ticker)
        q_fin = getattr(tkr, "quarterly_financials", pd.DataFrame())
        fin = getattr(tkr, "financials", pd.DataFrame())
        bs = getattr(tkr, "balance_sheet", pd.DataFrame())
        cf = getattr(tkr, "cashflow", pd.DataFrame())
        info = getattr(tkr, "info", {}) or {}

        revenue_series = _clean_series(q_fin.loc["Total Revenue"]) if "Total Revenue" in q_fin.index else _clean_series(fin.loc["Total Revenue"]) if "Total Revenue" in fin.index else []
        ebitda_series = _clean_series(fin.loc["Ebitda"]) if "Ebitda" in fin.index else []
        margin_series: List[Tuple[str, float]] = []
        if revenue_series and ebitda_series:
            rev_dict = dict(revenue_series)
            for period, ebitda in ebitda_series:
                rev = rev_dict.get(period)
                if rev:
                    margin_series.append((period, ebitda / rev))
        fcf_series: List[Tuple[str, float]] = []
        if "Total Cash From Operating Activities" in cf.index and "Capital Expenditures" in cf.index:
            ocf = _clean_series(cf.loc["Total Cash From Operating Activities"])
            capex = _clean_series(cf.loc["Capital Expenditures"])
            ocf_dict = dict(ocf)
            for period, cap in capex:
                if period in ocf_dict:
                    fcf_series.append((period, ocf_dict[period] + cap))

        net_debt = None
        if "Total Debt" in bs.index and "Cash" in bs.index:
            debt_series = _clean_series(bs.loc["Total Debt"])
            cash_series = _clean_series(bs.loc["Cash"])
            if debt_series and cash_series:
                net_debt = debt_series[-1][1] - cash_series[-1][1]

        shares = _safe_float(info.get("sharesOutstanding"))
        
        # Extract latest values for table
        latest_rev = revenue_series[-1][1] if revenue_series else _safe_float(info.get("totalRevenue"))
        latest_ebitda = ebitda_series[-1][1] if ebitda_series else _safe_float(info.get("ebitda"))
        latest_ni = _safe_float(info.get("netIncomeToCommon"))
        
        gross_margin = _safe_float(info.get("grossMargins"))
        op_margin = _safe_float(info.get("operatingMargins"))
        net_margin = _safe_float(info.get("profitMargins"))
        
        total_cash = _safe_float(info.get("totalCash"))
        total_debt = _safe_float(info.get("totalDebt"))
        total_assets = _clean_series(bs.loc["Total Assets"])[-1][1] if "Total Assets" in bs.index else None
        total_liab = _clean_series(bs.loc["Total Liabilities Net Minority Interest"])[-1][1] if "Total Liabilities Net Minority Interest" in bs.index else None
        equity = _clean_series(bs.loc["Stockholders Equity"])[-1][1] if "Stockholders Equity" in bs.index else None

        logger.info(f"Loaded financial snapshot for {ticker} in {time.time() - start_time:.2f}s")
        return FinancialSnapshot(
            revenue_series=revenue_series,
            ebitda_series=ebitda_series,
            margin_series=margin_series,
            fcf_series=fcf_series,
            net_debt=net_debt,
            shares_outstanding=shares,
            latest_revenue=latest_rev,
            latest_ebitda=latest_ebitda,
            latest_net_income=latest_ni,
            gross_margin=gross_margin,
            operating_margin=op_margin,
            net_margin=net_margin,
            total_cash=total_cash,
            total_debt=total_debt,
            total_assets=total_assets,
            total_liabilities=total_liab,
            equity=equity
        )
    except Exception as e:
        logger.error(f"Error loading financial snapshot for {ticker}: {e}")
        # Return empty snapshot on error
        return FinancialSnapshot([], [], [], [], None, None, None, None, None, None, None, None, None, None, None, None, None)


def load_valuation_snapshot(ticker: str, financials: FinancialSnapshot) -> ValuationSnapshot:
    tkr = yf.Ticker(ticker)
    info: Dict[str, float] = {}
    try:
        info = tkr.get_info()
    except Exception:
        info = getattr(tkr, "info", {}) or {}
    fast_info = getattr(tkr, "fast_info", {}) or {}

    price = _safe_float(fast_info.get("last_price") or info.get("currentPrice"))
    market_cap = _safe_float(fast_info.get("market_cap") or info.get("marketCap"))
    pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    ebitda = financials.latest_ebitda or _safe_float(info.get("ebitda"))

    enterprise_value = _safe_float(info.get("enterpriseValue"))
    if enterprise_value is None and market_cap is not None and financials.net_debt is not None:
        enterprise_value = market_cap + financials.net_debt

    ev_ebitda = None
    if enterprise_value and ebitda:
        ev_ebitda = enterprise_value / ebitda if ebitda != 0 else None

    fcf_yield = None
    if financials.fcf_series and market_cap:
        fcf = financials.fcf_series[-1][1]
        fcf_yield = fcf / market_cap

    growth = _safe_float(info.get("revenueGrowth"))
    div_yield = _safe_float(info.get("dividendYield"))

    return ValuationSnapshot(
        price=price,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        pe=pe,
        ev_ebitda=ev_ebitda,
        fcf_yield=fcf_yield,
        growth=growth,
        dividend_yield=div_yield
    )


def peer_list(sector: str, ticker: str) -> List[str]:
    sector_peers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Communication Services": ["GOOGL", "META", "NFLX", "TMUS"],
        "Consumer Cyclical": ["AMZN", "HD", "LOW", "NKE", "TSLA", "F", "GM"],
        "Consumer Defensive": ["PG", "KO", "PEP", "WMT"],
        "Energy": ["XOM", "CVX", "COP", "SLB"],
        "Financial Services": ["JPM", "BAC", "WFC", "GS"],
        "Healthcare": ["JNJ", "UNH", "PFE", "MRK"],
        "Industrials": ["CAT", "HON", "GE", "DE"],
        "Real Estate": ["PLD", "AMT", "EQIX", "SPG"],
        "Utilities": ["NEE", "DUK", "SO", "AEP"],
        "Basic Materials": ["LIN", "APD", "SHW", "FCX"],
    }
    peers = sector_peers.get(sector, ["SPY", "QQQ", "DIA"])
    peers = [p for p in peers if p.upper() != ticker.upper()]
    return peers[:4]


def load_peers_metrics(ticker: str, sector: str, financials: FinancialSnapshot) -> Dict[str, ValuationSnapshot]:
    start_time = time.time()
    peers = peer_list(sector, ticker)
    logger.info(f"Loading metrics for {len(peers)} peers: {peers}...")
    
    metrics: Dict[str, ValuationSnapshot] = {}
    
    def fetch_peer(peer_ticker: str) -> Optional[Tuple[str, ValuationSnapshot]]:
        try:
            peer_fin = load_financial_snapshot(peer_ticker)
            val = load_valuation_snapshot(peer_ticker, peer_fin)
            return peer_ticker, val
        except Exception as e:
            logger.warning(f"Failed to load peer {peer_ticker}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_peer = {executor.submit(fetch_peer, p): p for p in peers}
        for future in concurrent.futures.as_completed(future_to_peer):
            result = future.result()
            if result:
                metrics[result[0]] = result[1]
                
    logger.info(f"Loaded peer metrics in {time.time() - start_time:.2f}s")
    return metrics


def build_scenarios(
    ticker: str,
    valuation: ValuationSnapshot,
    financials: FinancialSnapshot,
    peers: Dict[str, ValuationSnapshot],
    expected_return: float,
) -> List[ScenarioCase]:
    last_price = valuation.price or 0.0
    ebitda = financials.latest_ebitda
    peer_ev_ebitda_values = [m.ev_ebitda for m in peers.values() if m.ev_ebitda]
    peer_multiple = np.median(peer_ev_ebitda_values) if peer_ev_ebitda_values else valuation.ev_ebitda or 10.0
    current_multiple = valuation.ev_ebitda or peer_multiple

    def implied_price(multiplier: float) -> Optional[float]:
        if not ebitda or not financials.shares_outstanding:
            return None
        net_debt = financials.net_debt or 0.0
        ev = ebitda * multiplier + net_debt
        return ev / financials.shares_outstanding

    # Base Case: Converge to peer median or maintain current trajectory
    base_multiple = (current_multiple + peer_multiple) / 2
    base_price = implied_price(base_multiple) or last_price * (1 + expected_return)
    base_reasoning = f"Valuation converges toward peer median of {peer_multiple:.1f}x EV/EBITDA."

    # Bull Case: Multiple expansion + growth premium
    bull_multiple = max(peer_multiple * 1.2, current_multiple * 1.15)
    bull_price = implied_price(bull_multiple) or last_price * (1 + expected_return + 0.15)
    bull_reasoning = "Multiple expansion driven by growth acceleration and margin improvement."

    # Bear Case: Multiple compression or execution miss
    bear_multiple = min(peer_multiple * 0.8, current_multiple * 0.85)
    bear_price = implied_price(bear_multiple) or last_price * (1 + expected_return - 0.20)
    bear_reasoning = "Multiple compression due to macro headwinds or execution risks."

    cases = [
        ScenarioCase("Bull", float(bull_price), float(bull_price / last_price - 1) if last_price else 0.0, 0.25, bull_reasoning),
        ScenarioCase("Base", float(base_price), float(base_price / last_price - 1) if last_price else expected_return, 0.50, base_reasoning),
        ScenarioCase("Bear", float(bear_price), float(bear_price / last_price - 1) if last_price else -0.2, 0.25, bear_reasoning),
    ]
    return cases


# --------------------------- Rendering ---------------------------

def _render_financial_table(fin: FinancialSnapshot) -> str:
    rows = [
        ("Total Revenue", _fmt_curr(fin.latest_revenue)),
        ("EBITDA", _fmt_curr(fin.latest_ebitda)),
        ("Net Income", _fmt_curr(fin.latest_net_income)),
        ("Gross Margin", _fmt_pct(fin.gross_margin)),
        ("Operating Margin", _fmt_pct(fin.operating_margin)),
        ("Net Margin", _fmt_pct(fin.net_margin)),
    ]
    
    latex = "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Metric} & \\textbf{Value} \\\\\n\\midrule\n"
    for label, val in rows:
        latex += f"{label} & {val} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    return latex

def _render_balance_sheet_table(fin: FinancialSnapshot) -> str:
    rows = [
        ("Total Cash", _fmt_curr(fin.total_cash)),
        ("Total Debt", _fmt_curr(fin.total_debt)),
        ("Net Debt", _fmt_curr(fin.net_debt)),
        ("Total Assets", _fmt_curr(fin.total_assets)),
        ("Total Liabilities", _fmt_curr(fin.total_liabilities)),
        ("Stockholders Equity", _fmt_curr(fin.equity)),
        ("Shares Outstanding", _fmt_curr(fin.shares_outstanding, short=True).replace("$", "")), # Hack to remove $ from shares
    ]
    
    latex = "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Item} & \\textbf{Value} \\\\\n\\midrule\n"
    for label, val in rows:
        latex += f"{label} & {val} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    return latex

def _render_valuation_table(val: ValuationSnapshot) -> str:
    rows = [
        ("Price", _fmt_curr(val.price)),
        ("Market Cap", _fmt_curr(val.market_cap)),
        ("Enterprise Value", _fmt_curr(val.enterprise_value)),
        ("P/E Ratio", f"{val.pe:.1f}x" if val.pe else "-"),
        ("EV/EBITDA", f"{val.ev_ebitda:.1f}x" if val.ev_ebitda else "-"),
        ("FCF Yield", _fmt_pct(val.fcf_yield)),
        ("Dividend Yield", _fmt_pct(val.dividend_yield)),
    ]
    
    latex = "\\begin{tabular}{lr}\n\\toprule\n\\textbf{Metric} & \\textbf{Value} \\\\\n\\midrule\n"
    for label, val in rows:
        latex += f"{label} & {val} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}"
    return latex


def build_latex_report(final_state: dict, selections: dict, decision: Optional[str], report_dir: Path) -> str:
    total_start = time.time()
    logger.info("Starting LaTeX report generation...")
    
    ticker = selections["ticker"]
    analysis_date = selections["analysis_date"]
    lookback = selections.get("lookback_days", 180)

    price = load_price_snapshot(ticker, analysis_date, lookback)
    financials = load_financial_snapshot(ticker)
    valuation = load_valuation_snapshot(ticker, financials)

    sector = "General"
    try:
        info = yf.Ticker(ticker).get_info()
        sector = info.get("sector", "General") or "General"
        company_name = info.get("longName", ticker)
        industry = info.get("industry", "")
        country = info.get("country", "")
    except Exception:
        company_name, industry, country = ticker, "", ""

    peers = load_peers_metrics(ticker, sector, financials)

    expected_return = price.one_year_return if price.one_year_return is not None else 0.08
    horizon_months = 12
    variant_view = "Multiple discount vs peers" if valuation.ev_ebitda and peers and any((valuation.ev_ebitda or 0) < (p.ev_ebitda or 0) for p in peers.values()) else "Growth durability versus market expectations"

    scenarios = build_scenarios(ticker, valuation, financials, peers, expected_return)
    
    # Chart snippets
    price_chart_data = list(zip(price.dates, price.closes))[-90:] # Last 90 days
    price_chart = _pgf_time_series(price_chart_data, f"{ticker} Price Trend (Last 90 Days)", "Price ($)")
    
    rev_data, rev_unit = _scale_series(financials.revenue_series[-8:])
    revenue_chart = _pgf_time_series(rev_data, "Quarterly Revenue", f"Revenue {rev_unit}")
    
    margin_chart = _pgf_time_series(financials.margin_series[-8:], "EBITDA Margin Trend", "Margin")
    
    peer_labels = list(peers.keys()) + [ticker]
    peer_ev_vals = [p.ev_ebitda or 0 for p in peers.values()] + [valuation.ev_ebitda or 0]
    valuation_chart = _pgf_bar_chart(peer_labels, peer_ev_vals, "EV/EBITDA Comparison", "Multiple (x)")

    # Technical Charts - Modular inclusion
    rsi_chart = ""
    macd_chart = ""
    include_ta = False
    
    # Simple heuristic: include TA if RSI is overbought/oversold or MACD is divergent
    if price.rsi:
        last_rsi = price.rsi[-1]
        if last_rsi > 65 or last_rsi < 35:
            include_ta = True
            rsi_data = list(zip(price.dates, price.rsi))[-90:]
            rsi_chart = _pgf_time_series(rsi_data, "RSI (14)", "RSI")
        
    if price.macd and price.macd_signal:
        if abs(price.macd[-1] - price.macd_signal[-1]) > 0.5: # Arbitrary threshold for "meaningful"
            include_ta = True
            macd_data = list(zip(price.dates, price.macd))[-90:]
            signal_data = list(zip(price.dates, price.macd_signal))[-90:]
            macd_chart = _pgf_time_series(
                macd_data, 
                "MACD (12, 26, 9)", 
                "Value", 
                extra_coords=signal_data,
                legend_labels=["MACD", "Signal"]
            )

    # Content extraction and cleaning
    market_report = _clean_agent_text(final_state.get("market_report") or "Market context summarized by agents unavailable.")
    fundamentals_report = _clean_agent_text(final_state.get("fundamentals_report") or "Fundamentals summary pending from agent.")
    news_report = _clean_agent_text(final_state.get("news_report") or "News and catalysts collected by agents.")
    sentiment_report = _clean_agent_text(final_state.get("sentiment_report") or "Sentiment sample not provided.")
    pm_decision = _clean_agent_text(final_state.get("final_trade_decision") or (decision or "Hold"))

    # Extract sentiment score if present
    sentiment_score_match = re.search(r"(?:Sentiment Score|Score):\s*(\d+(?:/\d+)?)", sentiment_report, re.IGNORECASE)
    sentiment_score_display = ""
    if sentiment_score_match:
        sentiment_score_display = f"\\textbf{{Sentiment Score:}} {sentiment_score_match.group(1)} \\\\"

    thesis_points: List[str] = []
    investment_plan = final_state.get("investment_plan") or ""
    for line in investment_plan.split("\n"):
        if line.strip().startswith("-"):
            thesis_points.append(line.strip("- "))
        if len(thesis_points) >= 8:
            break
    if not thesis_points:
        thesis_points = [
            "Underappreciated operating leverage as revenue scales",
            "Valuation discount to peer median on EV/EBITDA",
            "Upcoming catalysts expected to unlock sentiment reset",
        ]

    risk_state = final_state.get("risk_debate_state", {}) or {}
    risk_notes = _clean_agent_text(risk_state.get("history") or "Risk committee notes not captured.")

    # Scenario table with reasoning
    def table_row(label: str, value: str, reasoning: str) -> str:
        return rf"{_latex_escape(label)} & {_latex_escape(value)} & {_latex_escape(reasoning)} \\ \hline"

    scenario_rows = "\n".join(
        table_row(
            f"{case.name} ({case.probability*100:.0f}% prob)",
            f"${case.price:,.2f} | {_fmt_pct(case.return_pct)}",
            case.reasoning
        )
        for case in scenarios
    )
    
    logger.info(f"Report generation completed in {time.time() - total_start:.2f}s")

    # Peers table
    peer_rows = []
    for name, snap in peers.items():
        pe_display = f"{snap.pe:.1f}" if snap.pe else "-"
        ev_ebitda_display = f"{snap.ev_ebitda:.1f}" if snap.ev_ebitda else "-"
        row = "{} & {} & {} & {} \\\\".format(
            _latex_escape(name), pe_display, ev_ebitda_display, _fmt_pct(snap.fcf_yield)
        )
        peer_rows.append(row)
    peer_table = "\n".join(peer_rows) if peer_rows else "\\textit{Peer metrics unavailable.}"

    ev_ebitda_display = f"{valuation.ev_ebitda:.1f}" if valuation.ev_ebitda is not None else "-"
    peer_median_display = f"{np.median([p.ev_ebitda for p in peers.values() if p.ev_ebitda]):.1f}" if peers else "-"

    latex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}
\\usepackage{{float}}
\\usepackage{{titlesec}}
\\usepackage{{pgfplots}}
\\usepackage{{parskip}}
\\pgfplotsset{{compat=1.18}}

% Custom section formatting
\\titleformat{{\\section}}
  {{\\normalfont\\Large\\bfseries\\color{{blue!40!black}}}}{{\\thesection}}{{1em}}{{}}
\\titleformat{{\\subsection}}
  {{\\normalfont\\large\\bfseries\\color{{gray!80!black}}}}{{\\thesubsection}}{{1em}}{{}}

\\title{{{{ { _latex_escape(company_name) } ( { _latex_escape(ticker) } ) Investment Memo }}}}
\\author{{{{TradingAgents Multi-LLM Desk}}}}
\\date{{{{{_latex_escape(analysis_date)}}}}}

\\begin{{document}}
\\maketitle
\\tableofcontents
\\newpage

\\section{{Executive Summary}}
\\textbf{{Recommendation:}} { _latex_escape(decision or 'Pending') }.\\\\
\\textbf{{Target Return:}} {_fmt_pct(expected_return)} over ~{horizon_months} months.\\\\
\\textbf{{Variant Perception:}} {_latex_escape(variant_view)}.\\\\

\\textbf{{Upside Drivers:}}
\\begin{{itemize}}
\\item {_latex_escape(market_report[:200])}...
\\end{{itemize}}

\\textbf{{Downside Risks:}}
\\begin{{itemize}}
\\item {_latex_escape(risk_notes[:200])}...
\\end{{itemize}}

\\textbf{{Valuation Check:}} Current EV/EBITDA is {ev_ebitda_display}x vs peer median {peer_median_display}x.

\\section{{Company Overview}}
\\textbf{{Business Description:}} 
\\begin{{quote}}
{ _latex_format_text(fundamentals_report) }
\\end{{quote}}

\\subsection{{Financial Performance}}
\\begin{{table}}[H]
\\centering
{_render_financial_table(financials)}
\\caption{{Key Financial Metrics (TTM/Latest)}}
\\end{{table}}

\\subsection{{Balance Sheet Highlights}}
\\begin{{table}}[H]
\\centering
{_render_balance_sheet_table(financials)}
\\caption{{Balance Sheet Summary}}
\\end{{table}}

\\subsection{{Recent Developments}}
{_latex_format_text(news_report)}

\\section{{Industry and Competitive Landscape}}
\\textbf{{Market Drivers:}}
{_latex_format_text(market_report)}

\\textbf{{Competitive Positioning:}}
{sentiment_score_display}
{_latex_format_text(sentiment_report)}

\\section{{Financial Analysis}}
\\textbf{{Revenue & Growth:}}
Recent run-rate revenue is {_fmt_curr(financials.latest_revenue)} with estimated growth of {_fmt_pct(valuation.growth)}.

\\textbf{{Profitability:}}
EBITDA margin trend is visualized below.

\\begin{{figure}}[H]
\\centering
{price_chart}
\\caption{{Price Action (Last 90 Days)}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
{revenue_chart}
\\caption{{Quarterly Revenue Trend}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
{margin_chart}
\\caption{{EBITDA Margin Trajectory}}
\\end{{figure}}

"""

    if include_ta:
        latex += f"""
\\section{{Technical Analysis}}
"""
        if rsi_chart:
            latex += f"""
\\begin{{figure}}[H]
\\centering
{rsi_chart}
\\caption{{Relative Strength Index (14)}}
\\end{{figure}}
"""
        if macd_chart:
            latex += f"""
\\begin{{figure}}[H]
\\centering
{macd_chart}
\\caption{{MACD Indicator}}
\\end{{figure}}
"""

    latex += f"""
\\section{{Valuation}}
\\begin{{table}}[H]
\\centering
{_render_valuation_table(valuation)}
\\caption{{Valuation Metrics}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
{valuation_chart}
\\caption{{Relative Valuation (EV/EBITDA)}}
\\end{{figure}}

\\subsection{{Scenario Analysis}}
\\begin{{longtable}}{{p{{0.25\\linewidth}}p{{0.25\\linewidth}}p{{0.40\\linewidth}}}}
\\toprule
\\textbf{{Scenario}} & \\textbf{{Outcome}} & \\textbf{{Reasoning}} \\\\
\\midrule
{scenario_rows}
\\bottomrule
\\end{{longtable}}

\\section{{Investment Thesis}}
\\begin{{itemize}}
{''.join(f"\\item {_latex_escape(pt)}\n" for pt in thesis_points)}
\\end{{itemize}}

\\section{{Risks and Disconfirming Evidence}}
{_latex_format_text(risk_notes)}

\\section{{Appendix}}
\\subsection{{Peer Comparison}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Company}} & \\textbf{{P/E}} & \\textbf{{EV/EBITDA}} & \\textbf{{FCF Yield}} \\\\
\\midrule
{peer_table}
\\bottomrule
\\end{{tabular}}

\\subsection{{Agent Trace}}
\\textbf{{Final Decision:}} {_latex_escape(pm_decision)}

\\end{{document}}
"""

    report_dir.mkdir(parents=True, exist_ok=True)
    return latex


def build_markdown_report(final_state: dict, selections: dict, decision: Optional[str]) -> str:
    """Lightweight Markdown companion to the LaTeX memo (no charts)."""
    ticker = selections["ticker"]
    analysis_date = selections["analysis_date"]
    lookback = selections.get("lookback_days", 180)

    price = load_price_snapshot(ticker, analysis_date, lookback)
    financials = load_financial_snapshot(ticker)
    valuation = load_valuation_snapshot(ticker, financials)

    sector = "General"
    company_name = ticker
    industry = ""
    country = ""
    try:
        info = yf.Ticker(ticker).get_info()
        sector = info.get("sector", "General") or "General"
        company_name = info.get("longName", ticker) or ticker
        industry = info.get("industry", "") or ""
        country = info.get("country", "") or ""
    except Exception:
        pass

    peers = load_peers_metrics(ticker, sector, financials)
    expected_return = price.one_year_return if price.one_year_return is not None else 0.08
    scenarios = build_scenarios(ticker, valuation, financials, peers, expected_return)
    prob_weighted = sum(c.price * c.probability for c in scenarios)
    pw_return = (prob_weighted / valuation.price - 1) if valuation.price else expected_return
    ev_ebitda_display = f"{valuation.ev_ebitda:.1f}" if valuation.ev_ebitda is not None else "-"

    # Content extraction
    market_report = final_state.get("market_report") or "Market context summarized by agents unavailable."
    fundamentals_report = final_state.get("fundamentals_report") or "Fundamentals summary pending from agent."
    news_report = final_state.get("news_report") or "News and catalysts collected by agents."
    sentiment_report = final_state.get("sentiment_report") or "Sentiment sample not provided."
    pm_decision = final_state.get("final_trade_decision") or (decision or "Hold")
    risk_state = final_state.get("risk_debate_state", {}) or {}
    risk_notes = risk_state.get("history") or "Risk committee notes not captured."

    thesis_points: List[str] = []
    investment_plan = final_state.get("investment_plan") or ""
    for line in investment_plan.split("\n"):
        if line.strip().startswith("-"):
            thesis_points.append(line.strip("- "))
        if len(thesis_points) >= 8:
            break
    if not thesis_points:
        thesis_points = [
            "Underappreciated operating leverage as revenue scales",
            "Valuation discount to peer median on EV/EBITDA",
            "Upcoming catalysts expected to unlock sentiment reset",
        ]

    lines: List[str] = []
    lines.append(f"# {company_name} ({ticker}) Investment Memo")
    lines.append(f"Date: {analysis_date}")
    lines.append("")
    lines.append(f"**Call:** {decision or 'Pending'}")
    lines.append(f"**Variant view:** Multiple discount vs peers")
    lines.append(f"**Probability-weighted return:** {_fmt_pct(pw_return)}")
    lines.append("")

    last_close = price.closes[-1] if price.closes else None
    lines.append("## Market Snapshot")
    lines.append(f"- Data window: {lookback} days ending {analysis_date}")
    lines.append(f"- Observations: {len(price.closes)} closing prices captured")
    lines.append(f"- Last close: {_fmt_curr(last_close)}")
    lines.append(f"- 3m return: {_fmt_pct(price.three_month_return)}")
    lines.append(f"- 1y return: {_fmt_pct(price.one_year_return)}")
    lines.append(f"- Volatility (ann.): {_fmt_pct(price.volatility)}")
    beta_display = f"{price.beta:.2f}" if price.beta is not None else "-"
    lines.append(f"- Beta vs SPY: {beta_display}")
    lines.append("")

    lines.append("## Fundamentals Snapshot")
    lines.append(f"- Revenue trend points: {len(financials.revenue_series)}")
    lines.append(f"- EBITDA series points: {len(financials.ebitda_series)}")
    lines.append(f"- Latest net debt: {_fmt_curr(financials.net_debt)}")
    lines.append(f"- Shares outstanding: {_fmt_curr(financials.shares_outstanding)}")
    lines.append("")

    lines.append("## Valuation")
    lines.append(f"- Price: {_fmt_curr(valuation.price)}")
    lines.append(f"- Market cap: {_fmt_curr(valuation.market_cap)}")
    lines.append(f"- Enterprise value: {_fmt_curr(valuation.enterprise_value)}")
    lines.append(f"- P/E: {_fmt_pct(valuation.pe)}")
    lines.append(f"- EV/EBITDA: {ev_ebitda_display}")
    lines.append(f"- FCF yield: {_fmt_pct(valuation.fcf_yield)}")
    lines.append("")

    lines.append("## Scenarios")
    lines.append("| Case | Price | Return | Probability |")
    lines.append("| --- | --- | --- | --- |")
    for case in scenarios:
        lines.append(f"| {case.name} | ${case.price:,.2f} | {_fmt_pct(case.return_pct)} | {case.probability*100:.0f}% |")
    lines.append("")

    if peers:
        lines.append("## Peer Snapshot")
        lines.append("| Peer | P/E | EV/EBITDA | FCF Yield |")
        lines.append("| --- | --- | --- | --- |")
        for name, snap in peers.items():
            pe_display = f"{snap.pe:.1f}" if snap.pe else "-"
            ev_ebitda_display = f"{snap.ev_ebitda:.1f}" if snap.ev_ebitda else "-"
            lines.append(f"| {name} | {pe_display} | {ev_ebitda_display} | {_fmt_pct(snap.fcf_yield)} |")
        lines.append("")

    lines.append("## Qualitative Takeaways")
    lines.append(f"### Market Context\n{market_report}\n")
    lines.append(f"### Fundamentals\n{fundamentals_report}\n")
    lines.append(f"### News & Catalysts\n{news_report}\n")
    lines.append(f"### Sentiment\n{sentiment_report}\n")
    lines.append(f"### Risks\n{risk_notes}\n")
    lines.append("")

    lines.append("## Thesis Bullets")
    for pt in thesis_points:
        lines.append(f"- {pt}")
    lines.append("")

    lines.append("## Decision Trail")
    lines.append(f"- Portfolio decision: {pm_decision}")
    lines.append(f"- Industry: {industry or 'N/A'}; Sector: {sector}; Country: {country or 'N/A'}")
    lines.append("")
    lines.append("_Charts are rendered in the LaTeX report via PGFPlots; this Markdown file mirrors the same underlying data without embedded figures._")

    return "\n".join(lines)
