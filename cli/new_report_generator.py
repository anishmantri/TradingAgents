import json
from typing import Dict, Any, List
from datetime import datetime
from cli.schema import (
    InvestmentMemo,
    FrontMatter,
    ExecutiveSummary,
    ReviewPeriodOverview,
    BusinessStrategy,
    ThesisDrivers,
    ValuationScenarios,
    RisksFalsification,
    CatalystsMonitoring,
    AlternativeViews,
    ThesisPillar,
    RiskRewardScenario,
    PriceStats,
    FundamentalMetric,
    Event,
    BusinessOverview,
    Segment,
    CompetitiveContext,
    CapitalAllocation,
    DetailedThesisPillar,
    ValuationMetric,
    Scenario,
    Risk,
    FalsificationCondition,
    Catalyst,
    MonitoringPlan,
    AlternativeView
)
from cli.latex_utils import escape_latex, format_number, format_currency, format_percentage, format_large_number

def generate_pgf_plot(data: List[Dict[str, Any]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> str:
    """Generates a PGFPlot string from data."""
    # Placeholder for PGFPlot generation logic
    # In a real implementation, this would generate the LaTeX code for the plot
    return f"% PGFPlot: {title} ({xlabel} vs {ylabel})"

def construct_investment_memo(final_state: Dict[str, Any], selections: Dict[str, Any]) -> InvestmentMemo:
    """Constructs the InvestmentMemo object from the final state."""
    
    ticker = final_state.get("company_of_interest", "UNKNOWN")
    date_str = final_state.get("trade_date", datetime.now().strftime("%Y-%m-%d"))
    
    # 1. Front Matter
    front_matter = FrontMatter(
        company_name=ticker, # Ideally full name, but ticker is what we have
        ticker=ticker,
        recommendation=final_state.get("final_trade_decision", "HOLD").split("\n")[0], # Simple extraction
        target_price="N/A", # Needs to be extracted or estimated
        current_price="N/A", # Needs to be extracted
        upside_downside="N/A",
        asset="Equity",
        region_sector="Global/Tech", # Placeholder
        as_of_date=date_str,
        review_period="L12M",
        intended_use="Internal Investment Review",
        illustrative_view="Long Term",
        confidence="Medium"
    )

    # 2. Executive Summary (from Research Manager)
    exec_summary_data = final_state.get("executive_summary", {})
    
    # Ensure we have at least 3 thesis pillars to satisfy schema
    pillars_data = exec_summary_data.get("thesis_pillars", [])
    while len(pillars_data) < 3:
        pillars_data.append({"pillar": "Data Missing", "description": "Thesis pillar could not be generated."})

    executive_summary = ExecutiveSummary(
        investment_summary=exec_summary_data.get("investment_summary", "Investment summary not provided."),
        thesis_pillars=[ThesisPillar(**p) for p in pillars_data],
        variant_view=exec_summary_data.get("variant_view", "Variant view not provided."),
        risk_reward_table=[RiskRewardScenario(**s) for s in exec_summary_data.get("risk_reward_scenarios", [])]
    )

    # 3. Review Period Overview (from Market & News Analysts)
    market_data = final_state.get("market_data", {})
    news_data = final_state.get("news_data", {})
    fundamentals_data = final_state.get("fundamentals_data", {})
    
    price_stats = PriceStats(**market_data.get("price_stats", {"period": "N/A", "return_pct": 0.0, "volatility": 0.0, "max_drawdown": 0.0}))
    
    fundamental_evolution = [FundamentalMetric(**m) for m in fundamentals_data.get("fundamental_evolution", [])]
    event_timeline = [Event(**e) for e in news_data.get("key_events", [])]

    review_period = ReviewPeriodOverview(
        price_stats=price_stats,
        fundamental_evolution=fundamental_evolution,
        event_timeline=event_timeline,
        attribution_narrative="Review period attribution narrative not provided." # Placeholder
    )

    # 4. Business & Strategic Position (from Fundamentals Analyst)
    biz_data = fundamentals_data.get("business_strategy", {})
    business_strategy = BusinessStrategy(
        overview=BusinessOverview(**biz_data.get("overview", {"description": "N/A", "segments": [], "revenue_drivers": "N/A"})),
        competitive_context=CompetitiveContext(**biz_data.get("competitive_context", {
            "position": "N/A", "moat_trend": "N/A", "competitors": [],
            "market_structure": "N/A", "moat_analysis": "N/A", "structural_trends": "N/A"
        })),
        capital_allocation=CapitalAllocation(**biz_data.get("capital_allocation", {
            "capital_structure": "N/A", "allocation_track_record": "N/A", "governance_issues": "None"
        }))
    )

    # 5. Thesis & Drivers (from Bull Researcher)
    bull_data = final_state.get("investment_debate_state", {}).get("bull_data", {})
    thesis_drivers = ThesisDrivers(
        pillars=[DetailedThesisPillar(**p) for p in bull_data.get("thesis_pillars", [])]
    )

    # 6. Valuation & Scenarios (Placeholder / Needs Logic)
    # Currently we don't have a dedicated Valuation Analyst outputting this structure.
    # We might need to infer it or leave it as placeholders.
    valuation_scenarios = ValuationScenarios(
        current_valuation=[ValuationMetric(metric="P/E", value=0.0, history_avg=0.0, peer_avg=0.0)], # Placeholders
        valuation_frameworks="DCF and Relative Valuation", # Placeholder
        scenario_analysis=[
            Scenario(name="Base Case", revenue_growth="5%", ebitda_margin="20%", eps_fcf="10.0", implied_valuation=150.0, upside_downside=0.0, probability=0.5),
            Scenario(name="Bull Case", revenue_growth="10%", ebitda_margin="25%", eps_fcf="12.0", implied_valuation=180.0, upside_downside=20.0, probability=0.25),
            Scenario(name="Bear Case", revenue_growth="0%", ebitda_margin="15%", eps_fcf="8.0", implied_valuation=120.0, upside_downside=-20.0, probability=0.25)
        ],
        interpretation="Valuation appears attractive relative to growth." # Placeholder
    )

    # 7. Risks & Falsification (from Risk Manager)
    risk_data = final_state.get("risks_falsification", {})
    risks_falsification = RisksFalsification(
        fundamental_risks=[Risk(**r) for r in risk_data.get("risks", [])],
        structural_risks=[], # Placeholder
        falsification_conditions=[FalsificationCondition(**c) for c in risk_data.get("falsification_criteria", [])]
    )

    # 8. Catalysts & Monitoring (from News Analyst & Risk Manager)
    monitoring_data = final_state.get("monitoring_plan", {})
    # Map old keys to new schema if needed
    if monitoring_data:
        if "leading_indicators" in monitoring_data and "kpis" not in monitoring_data:
            monitoring_data["kpis"] = monitoring_data.pop("leading_indicators")
        if "data_sources" not in monitoring_data:
            monitoring_data["data_sources"] = ["N/A"]
        if "frequency" not in monitoring_data:
            monitoring_data["frequency"] = "Quarterly"
        if "triggers" not in monitoring_data:
            monitoring_data["triggers"] = "N/A"

    catalysts_monitoring = CatalystsMonitoring(
        upcoming_catalysts=[Catalyst(**c) for c in news_data.get("upcoming_catalysts", [])],
        monitoring_plan=MonitoringPlan(**monitoring_data) if monitoring_data else MonitoringPlan(kpis=[], data_sources=[], frequency="N/A", triggers="N/A")
    )

    # 9. Alternative Views (from Bear Researcher)
    bear_data = final_state.get("investment_debate_state", {}).get("bear_data", {})
    # Construct placeholders if data is missing
    default_view = {
        "case_name": "N/A", "summary": "N/A", "key_assumptions": "N/A", "support_from_review": "N/A"
    }
    
    bear_view_data = bear_data.get("alternative_view", default_view)
    # We don't have explicit bull alternative view from Bull Researcher usually, so use placeholder or extract
    bull_view_data = default_view 

    alternative_views = AlternativeViews(
        bull_case=AlternativeView(**bull_view_data),
        bear_case=AlternativeView(**bear_view_data),
        major_unknowns=["N/A"] # Placeholder
    )

    return InvestmentMemo(
        front_matter=front_matter,
        executive_summary=executive_summary,
        review_period=review_period,
        business_strategy=business_strategy,
        thesis=thesis_drivers,
        valuation=valuation_scenarios,
        risks=risks_falsification,
        catalysts=catalysts_monitoring,
        alternatives=alternative_views,
        data_sources=["Yahoo Finance", "NewsAPI", "Reddit"] # Placeholder
    )

def render_latex_report(memo: InvestmentMemo) -> str:
    """Renders the InvestmentMemo into a LaTeX string."""
    
    latex = f"""
\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{pgfplots}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\geometry{{a4paper, margin=1in}}

\\title{{Investment Memo: {escape_latex(memo.front_matter.company_name)} ({escape_latex(memo.front_matter.ticker)})}}
\\author{{Trading Agents AI}}
\\date{{{escape_latex(memo.front_matter.as_of_date)}}}

\\begin{{document}}

\\maketitle

\\section{{Executive Summary}}
\\textbf{{Recommendation:}} {escape_latex(memo.front_matter.recommendation)}

\\subsection{{Thesis Pillars}}
"""
    if memo.executive_summary.thesis_pillars:
        latex += "\\begin{itemize}\n"
        for pillar in memo.executive_summary.thesis_pillars:
            latex += f"\\item \\textbf{{{escape_latex(pillar.pillar)}}}: {escape_latex(pillar.description)}\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No thesis pillars identified.}\n"
    
    latex += """

\\section{Review Period Overview}
\\subsection{Price Stats}
"""
    latex += f"Return: {format_percentage(memo.review_period.price_stats.return_pct)} | Volatility: {format_number(memo.review_period.price_stats.volatility)} | Max Drawdown: {format_percentage(memo.review_period.price_stats.max_drawdown)}\n"

    latex += """
\\subsection{Fundamental Evolution}
\\begin{table}[H]
\\centering
\\begin{tabular}{l c c c p{6cm}}
\\toprule
Metric & Start & End & Change & Comment \\\\
\\midrule
"""
    for metric in memo.review_period.fundamental_evolution:
        latex += f"{escape_latex(metric.metric)} & {format_large_number(metric.start_value, currency=True)} & {format_large_number(metric.end_value, currency=True)} & {format_percentage(metric.change_pct)} & {escape_latex(metric.comment)} \\\\\n"
    
    latex += """
\\bottomrule
\\end{tabular}
\\end{table}

\\section{Business & Strategic Position}
"""
    latex += f"\\textbf{{Overview}}: {escape_latex(memo.business_strategy.overview.description)}\n\n"
    latex += f"\\textbf{{Competitive Position}}: {escape_latex(memo.business_strategy.competitive_context.market_structure)}\n"

    latex += """
\\section{Thesis & Drivers}
"""
    if memo.thesis.pillars:
        latex += "\\begin{itemize}\n"
        for pillar in memo.thesis.pillars:
            latex += f"\\item \\textbf{{{escape_latex(pillar.title)}}}: {escape_latex(pillar.claim)}\n"
            latex += f"  \\begin{{itemize}}\\item Evidence: {escape_latex(pillar.evidence_analysis)}\\end{{itemize}}\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No thesis drivers identified.}\n"

    latex += """
\\section{Valuation Scenarios}
"""
    # ... PGF Plot placeholder ...
    if memo.valuation.scenario_analysis:
        latex += "\\begin{itemize}\n"
        for scenario in memo.valuation.scenario_analysis:
            latex += f"\\item \\textbf{{{escape_latex(scenario.name)}}}: {format_currency(scenario.implied_valuation)} (Prob: {format_number(scenario.probability)})\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No valuation scenarios available.}\n"

    latex += """
\\section{Risks & Falsification}
\\subsection*{Key Risks}
"""
    if memo.risks.fundamental_risks:
        latex += "\\begin{itemize}\n"
        for risk in memo.risks.fundamental_risks:
            latex += f"\\item \\textbf{{{escape_latex(risk.category)}}}: {escape_latex(risk.description)} (Impact: {escape_latex(risk.impact)})\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No key risks identified.}\n"

    latex += """
\\subsection*{Falsification Criteria}
"""
    if memo.risks.falsification_conditions:
        latex += "\\begin{itemize}\n"
        for criteria in memo.risks.falsification_conditions:
            latex += f"\\item {escape_latex(criteria.condition)} (Threshold: {escape_latex(criteria.threshold)})\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No falsification criteria identified.}\n"

    latex += """
\\section{Catalysts & Monitoring}
\\subsection*{Upcoming Catalysts}
"""
    if memo.catalysts.upcoming_catalysts:
        latex += "\\begin{itemize}\n"
        for catalyst in memo.catalysts.upcoming_catalysts:
            latex += f"\\item \\textbf{{{escape_latex(catalyst.event)}}}: {escape_latex(catalyst.expected_impact)}\n"
        latex += "\\end{itemize}\n"
    else:
        latex += "\\textit{No upcoming catalysts identified.}\n"

    latex += """
\\section{Alternative Views}
\\begin{itemize}
"""
    latex += f"\\item \\textbf{{Bear Case}}: {escape_latex(memo.alternatives.bear_case.summary)}\n"
    latex += f"\\item \\textbf{{Bull Case}}: {escape_latex(memo.alternatives.bull_case.summary)}\n"
    latex += """
\\end{itemize}

\\end{document}
"""
    return latex
