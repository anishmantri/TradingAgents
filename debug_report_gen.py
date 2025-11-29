import sys
import os
from pathlib import Path
import logging
import time

# Add project root to path
sys.path.append(os.getcwd())

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from cli.report_generator import build_latex_report

# Mock data
final_state = {
    "market_report": "Market is doing well.",
    "fundamentals_report": "Company is strong.",
    "news_report": "Good news.",
    "sentiment_report": "Sentiment Score: 8/10",
    "investment_plan": "- Buy now\n- Hold later",
    "final_trade_decision": "Buy",
    "risk_debate_state": {"history": "Risk is low."},
}

selections = {
    "ticker": "AAPL", 
    "analysis_date": "2023-10-27",
    "lookback_days": 180
}

report_dir = Path("results/debug_test")
report_dir.mkdir(parents=True, exist_ok=True)

print("Starting report generation test...")
start = time.time()
try:
    build_latex_report(final_state, selections, "Buy", report_dir)
    print(f"Report generation finished in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
