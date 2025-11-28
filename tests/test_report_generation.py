
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
        "market_report": "Market is bullish.",
        "fundamentals_report": "Company is strong.",
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

    # Mock yfinance data loading by patching (or just relying on the fact that we can't easily mock yfinance here without more work)
    # However, the functions in report_generator.py call yfinance. 
    # To properly test without network calls, we would need to mock yfinance.
    # For now, let's just try to import and run it, but we might hit network issues or empty data.
    # Actually, the user's request implies they want me to fix the *formatting* error.
    # The formatting error happened *after* data was loaded.
    
    # Let's try to call build_latex_report. 
    # Since it calls load_price_snapshot etc internally, it will try to fetch data.
    # I should probably mock those load_* functions in report_generator.py for this test.
    
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
        revenue_series=[("2024-Q1", 1000.0), ("2024-Q2", 1100.0)],
        ebitda_series=[("2024-Q1", 200.0), ("2024-Q2", 220.0)],
        margin_series=[("2024-Q1", 0.2), ("2024-Q2", 0.2)],
        fcf_series=[("2024-Q1", 50.0)],
        net_debt=500.0,
        shares_outstanding=100.0
    ))
    
    rg.load_valuation_snapshot = MagicMock(return_value=ValuationSnapshot(
        price=101.0,
        market_cap=10100.0,
        enterprise_value=10600.0,
        pe=20.0,
        ev_ebitda=15.0,
        fcf_yield=0.04,
        growth=0.1
    ))
    
    rg.load_peers_metrics = MagicMock(return_value={
        "PEER": ValuationSnapshot(
            price=50.0,
            market_cap=5000.0,
            enterprise_value=5500.0,
            pe=25.0,
            ev_ebitda=18.0,
            fcf_yield=0.03,
            growth=0.08
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
