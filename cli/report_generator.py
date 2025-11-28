"""Generate a structured LaTeX report for the TradingAgents CLI.

The builder assembles a hedge-fund style memo with data pulled from yfinance
and the agent outputs. Graphs are rendered with PGFPlots inside LaTeX so the
output is a single self-contained ``.tex`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------- Data Containers ---------------------------


@dataclass
class PriceSnapshot:
    dates: List[str]
    closes: List[float]
    one_year_return: Optional[float]
    three_month_return: Optional[float]
    volatility: Optional[float]
    beta: Optional[float]


@dataclass
class FinancialSnapshot:
    revenue_series: List[Tuple[str, float]]
    ebitda_series: List[Tuple[str, float]]
    margin_series: List[Tuple[str, float]]
    fcf_series: List[Tuple[str, float]]
    net_debt: Optional[float]
    shares_outstanding: Optional[float]


@dataclass
class ValuationSnapshot:
    price: Optional[float]
    market_cap: Optional[float]
    enterprise_value: Optional[float]
    pe: Optional[float]
    ev_ebitda: Optional[float]
    fcf_yield: Optional[float]
    growth: Optional[float]


@dataclass
class ScenarioCase:
    name: str
    price: float
    return_pct: float
    probability: float


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


def _clean_series(series: pd.Series, limit: int = 6) -> List[Tuple[str, float]]:
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


def _pgf_time_series(coords: List[Tuple[str, float]], title: str, ylabel: str) -> str:
    if not coords:
        return "\\textit{Insufficient data for chart.}"
    xs = list(range(len(coords)))
    xticks = " ,".join(str(i) for i in xs)
    xticklabels = " ,".join(label for label, _ in coords)
    points = " ".join(f"({i},{y:.2f})" for i, (_, y) in zip(xs, coords))
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.95\\linewidth,height=6cm,grid=both,"
        f"title={{{{title}}}},ylabel={{{{ylabel}}}},xtick={{{{ {xticks} }}}},"
        f"xticklabels={{{{{xticklabels}}}}},xticklabel style={{rotate=45, anchor=east}},"
        "ymajorgrids,xmajorgrids,legend style={at={(0.02,0.98)},anchor=north west}]\n"
        f"\\addplot+[mark=*,thick,color=blue] coordinates {{{points}}};\n"
        "\\end{axis}\n\\end{tikzpicture}"
    )


def _pgf_bar_chart(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> str:
    if not labels or not values:
        return "\\textit{Insufficient data for chart.}"
    coords = " ".join(f"({i},{v:.2f})" for i, v in enumerate(values))
    xticks = " ,".join(str(i) for i in range(len(labels)))
    xticklabels = " ,".join(labels)
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[ybar,bar width=12pt,width=0.95\\linewidth,height=6cm,"
        "grid=both,enlarge x limits=0.15,"
        f"title={{{{title}}}},ylabel={{{{ylabel}}}},xtick={{{{ {xticks} }}}},"
        f"xticklabels={{{{{xticklabels}}}}},xticklabel style={{rotate=45, anchor=east}}]\n"
        f"\\addplot+[fill=blue!65] coordinates {{{coords}}};\n"
        "\\end{axis}\n\\end{tikzpicture}"
    )


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"{val*100:.1f}%"


def _fmt_curr(val: Optional[float]) -> str:
    if val is None:
        return "-"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.1f}M"
    return f"${val:,.0f}"


def _latex_escape(text: str) -> str:
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
    }
    for src, repl in replacements.items():
        text = text.replace(src, repl)
    return text


# --------------------------- Data Collection ---------------------------


def load_price_snapshot(ticker: str, analysis_date: str, lookback_days: int) -> PriceSnapshot:
    end_dt = pd.to_datetime(analysis_date)
    start_dt = end_dt - timedelta(days=max(lookback_days, 120))
    price_df = yf.download(ticker, start=start_dt, end=end_dt + timedelta(days=1), progress=False)
    if price_df.empty:
        return PriceSnapshot([], [], None, None, None, None)

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    price_df = price_df.sort_index()
    close_candidates = ["Adj Close", "Close", "adjclose", "adj_close", "close"]
    close_col = next((c for c in close_candidates if c in price_df.columns), None)
    if not close_col:
        return PriceSnapshot([], [], None, None, None, None)

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

    # Beta vs SPY
    spy_df = yf.download("SPY", start=start_dt, end=end_dt + timedelta(days=1), progress=False)
    beta = None
    if not spy_df.empty:
        spy_ret = spy_df["Close"].pct_change().dropna()
        beta = _compute_beta(returns, spy_ret)

    return PriceSnapshot(dates=dates, closes=list(closes.values), one_year_return=ret_1y, three_month_return=ret_3m, volatility=vol, beta=beta)


def load_financial_snapshot(ticker: str) -> FinancialSnapshot:
    tkr = yf.Ticker(ticker)
    q_fin = getattr(tkr, "quarterly_financials", pd.DataFrame())
    fin = getattr(tkr, "financials", pd.DataFrame())
    bs = getattr(tkr, "balance_sheet", pd.DataFrame())
    cf = getattr(tkr, "cashflow", pd.DataFrame())

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
    shares = None
    if "Total Debt" in bs.index and "Cash" in bs.index:
        debt_series = _clean_series(bs.loc["Total Debt"])
        cash_series = _clean_series(bs.loc["Cash"])
        if debt_series and cash_series:
            net_debt = debt_series[-1][1] - cash_series[-1][1]

    try:
        shares = _safe_float(getattr(tkr, "info", {}).get("sharesOutstanding"))
    except Exception:
        shares = None

    return FinancialSnapshot(
        revenue_series=revenue_series,
        ebitda_series=ebitda_series,
        margin_series=margin_series,
        fcf_series=fcf_series,
        net_debt=net_debt,
        shares_outstanding=shares,
    )


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
    ebitda = None
    if financials.ebitda_series:
        ebitda = financials.ebitda_series[-1][1]
    elif info.get("ebitda"):
        ebitda = _safe_float(info.get("ebitda"))

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

    return ValuationSnapshot(
        price=price,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        pe=pe,
        ev_ebitda=ev_ebitda,
        fcf_yield=fcf_yield,
        growth=growth,
    )


def peer_list(sector: str, ticker: str) -> List[str]:
    sector_peers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Communication Services": ["GOOGL", "META", "NFLX", "TMUS"],
        "Consumer Cyclical": ["AMZN", "HD", "LOW", "NKE"],
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
    peers = peer_list(sector, ticker)
    metrics: Dict[str, ValuationSnapshot] = {}
    for peer in peers:
        try:
            peer_fin = load_financial_snapshot(peer)
            metrics[peer] = load_valuation_snapshot(peer, peer_fin)
        except Exception:
            continue
    return metrics


def build_scenarios(
    ticker: str,
    valuation: ValuationSnapshot,
    financials: FinancialSnapshot,
    peers: Dict[str, ValuationSnapshot],
    expected_return: float,
) -> List[ScenarioCase]:
    last_price = valuation.price or 0.0
    ebitda = financials.ebitda_series[-1][1] if financials.ebitda_series else None
    peer_ev_ebitda_values = [m.ev_ebitda for m in peers.values() if m.ev_ebitda]
    peer_multiple = np.median(peer_ev_ebitda_values) if peer_ev_ebitda_values else valuation.ev_ebitda or 10.0

    def implied_price(multiplier: float) -> Optional[float]:
        if not ebitda or not financials.shares_outstanding:
            return None
        net_debt = financials.net_debt or 0.0
        ev = ebitda * multiplier + net_debt
        return ev / financials.shares_outstanding

    base_price = implied_price(peer_multiple) or last_price * (1 + expected_return)
    bull_price = implied_price(peer_multiple * 1.2) or last_price * (1 + expected_return + 0.1)
    bear_price = implied_price(max(peer_multiple * 0.8, peer_multiple - 2)) or last_price * (1 + expected_return - 0.15)

    cases = [
        ScenarioCase("Bull", float(bull_price), float(bull_price / last_price - 1) if last_price else 0.0, 0.25),
        ScenarioCase("Base", float(base_price), float(base_price / last_price - 1) if last_price else expected_return, 0.5),
        ScenarioCase("Bear", float(bear_price), float(bear_price / last_price - 1) if last_price else -0.2, 0.25),
    ]
    return cases


# --------------------------- Rendering ---------------------------


def build_latex_report(final_state: dict, selections: dict, decision: Optional[str], report_dir: Path) -> str:
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
    prob_weighted = sum(c.price * c.probability for c in scenarios)
    pw_return = (prob_weighted / valuation.price - 1) if valuation.price else expected_return

    # Chart snippets
    price_chart = _pgf_time_series(list(zip(price.dates[-24:], price.closes[-24:])), f"{ticker} price trend", "Price")
    revenue_chart = _pgf_time_series(financials.revenue_series[-6:], "Revenue trend", "$m")
    margin_chart = _pgf_time_series(financials.margin_series[-6:], "EBITDA margin", "Margin")
    peer_labels = list(peers.keys()) + [ticker]
    peer_ev_vals = [p.ev_ebitda or 0 for p in peers.values()] + [valuation.ev_ebitda or 0]
    peer_ev_clean = [p.ev_ebitda for p in peers.values() if p.ev_ebitda]
    peer_median_multiple = float(np.median(peer_ev_clean)) if peer_ev_clean else None
    valuation_chart = _pgf_bar_chart(peer_labels, peer_ev_vals, "EV/EBITDA vs peers", "x")

    def short(text: Optional[str], fallback: str, max_chars: int = 450) -> str:
        if not text:
            return fallback
        cleaned = text.strip().replace("\n", " ")
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")

    market_report = short(final_state.get("market_report"), "Market context summarized by agents unavailable.")
    fundamentals_report = short(final_state.get("fundamentals_report"), "Fundamentals summary pending from agent.")
    news_report = short(final_state.get("news_report"), "News and catalysts collected by agents.")
    sentiment_report = short(final_state.get("sentiment_report"), "Sentiment sample not provided.")
    pm_decision = short(final_state.get("final_trade_decision"), decision or "Hold")

    thesis_points: List[str] = []
    investment_plan = final_state.get("investment_plan") or ""
    for line in investment_plan.split("\n"):
        if line.strip().startswith("-"):
            thesis_points.append(line.strip("- "))
        if len(thesis_points) >= 5:
            break
    if not thesis_points:
        thesis_points = [
            "Underappreciated operating leverage as revenue scales",
            "Valuation discount to peer median on EV/EBITDA",
            "Upcoming catalysts expected to unlock sentiment reset",
        ]

    risk_state = final_state.get("risk_debate_state", {}) or {}
    risk_notes = short(risk_state.get("history"), "Risk committee notes not captured.")

    def table_row(label: str, value: str) -> str:
        return rf"{_latex_escape(label)} & {_latex_escape(value)} \\ \hline"

    # Scenario table
    scenario_rows = "\n".join(
        table_row(
            f"{case.name} ({case.probability*100:.0f}% prob)",
            f"${case.price:,.2f} | {_fmt_pct(case.return_pct)}",
        )
        for case in scenarios
    )

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
    peer_median_display = f"{peer_median_multiple:.1f}" if peer_median_multiple is not None else "-"

    latex = f"""
\\documentclass[12pt]{{article}}
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
\\pgfplotsset{{compat=1.18}}
\\title{{{{ { _latex_escape(company_name) } ( { _latex_escape(ticker) } ) Investment Memo }}}}
\\author{{{{TradingAgents Multi-LLM Desk}}}}
\\date{{{{{_latex_escape(analysis_date)}}}}}
\\begin{{document}}
\\maketitle
\\tableofcontents
\\newpage

\\section{{Executive Summary}}
\\textbf{{Call:}} { _latex_escape(decision or 'Pending') }.\\\newline
Expected return: {_fmt_pct(expected_return)} over ~{horizon_months} months; variant perception: {_latex_escape(variant_view)}.\\\newline
Primary upside drivers: {_latex_escape(market_report[:160])}.\\\newline
Primary downside drivers: {_latex_escape(risk_notes[:160])}.\\\newline
Why mispriced: market is not pricing {_latex_escape(variant_view.lower())}; current EV/EBITDA {ev_ebitda_display} vs peer median {peer_median_display}.

\\section{{Company Overview}}
{ _latex_escape(fundamentals_report) }\\\newline
Business segments & revenue mix: consolidated view; latest quarterly revenue {_fmt_curr(financials.revenue_series[-1][1] if financials.revenue_series else None)}.\\\newline
Geographic exposure: {_latex_escape(country or 'global markets')}.\\\newline
Recent strategic developments: {_latex_escape(news_report)}.

\\section{{Industry and Competitive Landscape}}
Core drivers: {_latex_escape(market_report)}.\\\newline
Competitive positioning: {_latex_escape(sentiment_report)}.\\\newline
Structural tailwinds/headwinds: {_latex_escape(news_report[:220])}.

\\section{{Catalysts (Upcoming and Medium-Term)}}
Hard catalysts: earnings cadence, product updates, regulatory checkpoints.\\\newline
Soft catalysts: capital allocation shifts, leadership signals, sentiment reversals.\\\newline
Expected timeline: next {horizon_months} months with market reaction tied to delivery vs guide.

\\section{{Financial Model Summary}}
\\textbf{{Revenue drivers}}: recent run-rate {_fmt_curr(financials.revenue_series[-1][1] if financials.revenue_series else None)}; growth {_fmt_pct(valuation.growth)}.\\\newline
\\textbf{{Margins}}: EBITDA margin trend below.\\\newline
\\textbf{{Cash generation}}: FCF {_fmt_curr(financials.fcf_series[-1][1] if financials.fcf_series else None)}; net debt {_fmt_curr(financials.net_debt)}; shares {_fmt_curr(financials.shares_outstanding)}.\\\newline
Sensitivity: scenario table below captures valuation delta to key drivers.

\\begin{{figure}}[H]
\\centering
{price_chart}
\\caption{{Price trend and momentum}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
{revenue_chart}
\\caption{{Revenue trajectory}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
{margin_chart}
\\caption{{EBITDA margin trend}}
\\end{{figure}}

\\section{{Valuation}}
Current price {_fmt_curr(valuation.price)}; market cap {_fmt_curr(valuation.market_cap)}; EV {_fmt_curr(valuation.enterprise_value)}.\\\newline
Multiples: P/E {_fmt_pct(valuation.pe)}; EV/EBITDA {ev_ebitda_display}; FCF yield {_fmt_pct(valuation.fcf_yield)}.\\\newline
Historical vs present: peer-adjusted comparison below.

\\begin{{figure}}[H]
\\centering
{valuation_chart}
\\caption{{EV/EBITDA vs peer set}}
\\end{{figure}}

\\begin{{longtable}}{{p{{0.45\\linewidth}}p{{0.45\\linewidth}}}}
\\toprule
Scenario & Outcome \\\n+\\midrule
{scenario_rows}
\\bottomrule
\\end{{longtable}}

\\section{{Investment Thesis}}
\\begin{{itemize}}
{''.join(f"\\item {_latex_escape(pt)}\n" for pt in thesis_points)}
\\end{{itemize}}

\\section{{Variant View}}
How we differ: {_latex_escape(variant_view)}; Street likely missing throughput on margins and durability of growth vs { _latex_escape(sector) }.\\\newline
Quantified spread vs consensus: probability-weighted return {_fmt_pct(pw_return)} relative to spot.

\\section{{Risks and Disconfirming Evidence}}
Key bear points: {_latex_escape(risk_notes)}.\\\newline
Early warnings: margin compression, revenue deceleration, tightening liquidity.\\\newline
Exit triggers: thesis break if bear case metrics persist or catalysts fail.

\\section{{Appendix}}
\\textbf{{Peers and comps}}\\\newline
{peer_table}

\\textbf{{Channel checks & filings}}\\\newline
{_latex_escape(news_report)}

\\textbf{{Agent trace}}\\\newline
Portfolio decision: {_latex_escape(pm_decision)}.

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

    def short(text: Optional[str], fallback: str, max_chars: int = 450) -> str:
        if not text:
            return fallback
        cleaned = str(text).strip().replace("\n", " ")
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")

    market_report = short(final_state.get("market_report"), "Market context summarized by agents unavailable.")
    fundamentals_report = short(final_state.get("fundamentals_report"), "Fundamentals summary pending from agent.")
    news_report = short(final_state.get("news_report"), "News and catalysts collected by agents.")
    sentiment_report = short(final_state.get("sentiment_report"), "Sentiment sample not provided.")
    pm_decision = short(final_state.get("final_trade_decision"), decision or "Hold")
    risk_state = final_state.get("risk_debate_state", {}) or {}
    risk_notes = short(risk_state.get("history"), "Risk committee notes not captured.")

    thesis_points: List[str] = []
    investment_plan = final_state.get("investment_plan") or ""
    for line in investment_plan.split("\n"):
        if line.strip().startswith("-"):
            thesis_points.append(line.strip("- "))
        if len(thesis_points) >= 5:
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
    lines.append(f"- Beta vs SPY: {price.beta:.2f if price.beta is not None else '-'}")
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
    lines.append(f"- Market context: {market_report}")
    lines.append(f"- Fundamentals: {fundamentals_report}")
    lines.append(f"- News & catalysts: {news_report}")
    lines.append(f"- Sentiment: {sentiment_report}")
    lines.append(f"- Risks: {risk_notes}")
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
