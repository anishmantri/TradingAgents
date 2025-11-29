import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from cli.schema import (
    InvestmentMemo, FrontMatter, ExecutiveSummary, ReviewPeriodOverview,
    BusinessStrategy, ThesisDrivers, ValuationScenarios, RisksFalsification,
    CatalystsMonitoring, AlternativeViews, PriceStats, FundamentalMetric,
    BusinessOverview, CompetitiveContext, CapitalAllocation, ThesisPillar,
    AlternativeView, RiskRewardScenario, MonitoringPlan
)
from cli.new_report_generator import render_latex_report

def test_report_generation():
    # Create dummy data
    memo = InvestmentMemo(
        front_matter=FrontMatter(
            company_name="Test Company Inc.",
            ticker="TEST",
            recommendation="BUY",
            target_price="150.00",
            current_price="100.00",
            upside_downside="50%",
            asset="Equity",
            region_sector="Tech",
            as_of_date="2025-11-29",
            review_period="L12M",
            intended_use="Test",
            illustrative_view="Long",
            confidence="High"
        ),
        executive_summary=ExecutiveSummary(
            investment_summary="This is a test summary with special chars: $ & % # _ { }",
            thesis_pillars=[
                ThesisPillar(pillar="Growth", description="High growth potential > 20%"),
                ThesisPillar(pillar="Margins", description="Expanding margins & cash flow"),
                ThesisPillar(pillar="Valuation", description="Attractive valuation relative to peers")
            ],
            variant_view="None",
            risk_reward_table=[]
        ),
        review_period=ReviewPeriodOverview(
            price_stats=PriceStats(period="L12M", return_pct=15.5, volatility=0.25, max_drawdown=-0.10),
            fundamental_evolution=[
                FundamentalMetric(metric="Revenue", start_value=1000000000, end_value=1200000000, change_pct=20.0, comment="Strong growth"),
                FundamentalMetric(metric="Net Income", start_value=-5000000, end_value=10000000, change_pct=300.0, comment="Turnaround")
            ],
            event_timeline=[],
            attribution_narrative="N/A"
        ),
        business_strategy=BusinessStrategy(
            overview=BusinessOverview(description="Test overview", segments=[], revenue_drivers="N/A"),
            competitive_context=CompetitiveContext(position="Leader", moat_trend="Stable", competitors=[], market_structure="Oligopoly", moat_analysis="N/A", structural_trends="N/A"),
            capital_allocation=CapitalAllocation(capital_structure="N/A", allocation_track_record="N/A", governance_issues="None")
        ),
        thesis=ThesisDrivers(pillars=[]),
        valuation=ValuationScenarios(current_valuation=[], valuation_frameworks="N/A", scenario_analysis=[], interpretation="N/A"),
        risks=RisksFalsification(fundamental_risks=[], structural_risks=[], falsification_conditions=[]),
        catalysts=CatalystsMonitoring(
            upcoming_catalysts=[], 
            monitoring_plan=MonitoringPlan(kpis=[], data_sources=[], frequency="Quarterly", triggers="N/A")
        ),
        alternatives=AlternativeViews(bull_case=AlternativeView(case_name="Bull", summary="Bullish", key_assumptions="N/A", support_from_review="N/A"), bear_case=AlternativeView(case_name="Bear", summary="Bearish", key_assumptions="N/A", support_from_review="N/A"), major_unknowns=[]),
        data_sources=[]
    )

    try:
        latex = render_latex_report(memo)
        print("LaTeX generation successful!")
        print(latex) # Debug output
        
        # Basic validation
        assert "\\section{Executive Summary}" in latex
        assert "Test Company Inc." in latex
        assert "15.5\\%" in latex # Formatted percentage
        assert "1.0B" in latex # Formatted large number
        assert "\\&" in latex # Escaped char
        assert "\\begin{itemize}" in latex
        
        # Check for empty itemize blocks (should not exist for empty lists)
        assert "\\section{Thesis & Drivers}" in latex
        assert "No thesis drivers identified" in latex
        
        print("Validation checks passed!")
        
        with open("test_report.tex", "w") as f:
            f.write(latex)
            
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_report_generation()
