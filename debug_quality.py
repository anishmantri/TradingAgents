import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

from cli.report_generator import _latex_format_text, _clean_agent_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_conversion():
    messy_input = """
    Market Analyst:
    Here is the report:
    
    ### Market Overview
    The market is **volatile** but showing signs of recovery.
    
    - Tech sector is up
    - Energy is down
      - Oil prices dropping
    
    Key risks:
    1. Inflation
    2. Geopolitics
    
    Analysis:
    The *trend* is your friend.
    
    | Metric | Value |
    |--------|-------|
    | P/E    | 20x   |
    | Growth | 5%    |
    
    Code snippet:
    ```python
    print("Hello")
    ```
    
    End of report.
    """
    
    print("--- Original Input ---")
    print(messy_input)
    
    cleaned = _clean_agent_text(messy_input)
    print("\n--- Cleaned Input ---")
    print(cleaned)
    
    latex = _latex_format_text(cleaned)
    print("\n--- LaTeX Output ---")
    print(latex)

if __name__ == "__main__":
    test_conversion()
