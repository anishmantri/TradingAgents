
import sys
import os
from pathlib import Path
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.report_generator import build_latex_report, build_markdown_report, FinancialSnapshot, ValuationSnapshot, PriceSnapshot

def test_report_generation():
    print("Testing report generation...")
    
    # Mock data
    final_state = {
        "market_report": "- Market is bullish.\n- Trend is up.",
        "fundamentals_report": "Company is strong.\n* Good management.\n* High margins.",
        "news_report": "New product launch.",
        "sentiment_report": "Positive sentiment.",
        "final_trade_decision": "BUY",
        "investment_plan": "- Strong growth\n- Good value\n- Catalyst incoming",
        "risk_debate_state": {
            "history": "Risks are manageable."
        }
    }
    
    selections = {
        "ticker": "TEST",
        "analysis_date": "2025-01-01",
        "lookback_days": 180,
    }
    
    decision = "BUY"
    report_dir = Path("test_results")
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir()

    import cli.report_generator as rg
    from unittest.mock import MagicMock
    
    # Mock data loaders
    rg.load_price_snapshot = MagicMock(return_value=PriceSnapshot(
        dates=["2024-01-01", "2024-01-02"],
        closes=[100.0, 101.0],
        one_year_return=0.1,
        three_month_return=0.05,
        volatility=0.2,
        beta=1.1
    ))
    
    rg.load_financial_snapshot = MagicMock(return_value=FinancialSnapshot(
        revenue_series=[("2024-Q1", 1.0e9), ("2024-Q2", 1.1e9)], # Billions
        ebitda_series=[("2024-Q1", 200.0e6), ("2024-Q2", 220.0e6)], # Millions
        margin_series=[("2024-Q1", 0.2), ("2024-Q2", 0.2)],
        fcf_series=[("2024-Q1", 50.0e6)],
        net_debt=500.0e6,
        shares_outstanding=100.0e6,
        latest_revenue=1.1e9,
        latest_ebitda=220.0e6,
        latest_net_income=150.0e6,
        gross_margin=0.4,
        operating_margin=0.25,
        net_margin=0.15,
        total_cash=100.0e6,
        total_debt=600.0e6,
        total_assets=2.0e9,
        total_liabilities=1.0e9,
        equity=1.0e9
    ))
    
    rg.load_valuation_snapshot = MagicMock(return_value=ValuationSnapshot(
        price=101.0,
        market_cap=10100.0,
        enterprise_value=10600.0,
        pe=20.0,
        ev_ebitda=15.0,
        fcf_yield=0.04,
        growth=0.1,
        dividend_yield=0.02
    ))
    
    rg.load_peers_metrics = MagicMock(return_value={
        "PEER": ValuationSnapshot(
            price=50.0,
            market_cap=5000.0,
            enterprise_value=5500.0,
            pe=25.0,
            ev_ebitda=18.0,
            fcf_yield=0.03,
            growth=0.08,
            dividend_yield=0.015
        )
    })
    
    rg.yf.Ticker = MagicMock() # Mock Ticker to avoid network calls in sector info
    
    try:
        latex = build_latex_report(final_state, selections, decision, report_dir)
        print("LaTeX report generated successfully.")
        print(f"Length: {len(latex)}")
        
        # Check for key sections
        assert "Executive Summary" in latex
        assert "Company Overview" in latex
        assert "Financial Analysis" in latex
        assert "Valuation" in latex
        assert "Investment Thesis" in latex
        
        # Check for the fix (no format error) and new format
        assert "Current EV/EBITDA is 15.0x" in latex
        
        # Check for newline fix
        assert "newline" not in latex.lower().replace("newline", "") # Ensure literal "newline" text is gone (except maybe in latex commands if any)
        # Better check: ensure we don't see "newline" as a standalone word in the text body
        assert " newline " not in latex
        
        # Check for content depth (no truncation)
        assert "Market is bullish." in latex
        assert "Company is strong." in latex
        
        # Check for PGFPlots enhancements
        assert "grid style={dashed,gray!30}" in latex
        assert "nodes near coords" in latex
        
        # Check for structured tables
        assert "\\begin{tabular}" in latex
        assert "Total Revenue" in latex
        assert "Gross Margin" in latex
        
        # Check for advanced text parsing
        # "### Market Context" -> \paragraph*{Market Context} or similar
        # We mocked market_report as "- Market is bullish." so let's check for itemize
        assert "\\begin{itemize}" in latex
        assert "\\item Market is bullish." in latex
        
        # Check for bolding in lists or text
        # We mocked fundamentals_report with "* Good management."
        assert "\\item Good management." in latex
        
        # Check for unit scaling in tables
        # Revenue was 1.0e9 -> $1.00B
        assert "1.00B" in latex or "1.0B" in latex

        # Test Markdown Report as well
        markdown = build_markdown_report(final_state, selections, decision)
        print("Markdown report generated successfully.")
        assert "Beta vs SPY: 1.10" in markdown
        assert "### Market Context" in markdown
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_report_generation()
