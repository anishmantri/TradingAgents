from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class ReportSection(BaseModel):
    title: str
    content: str


class CLIOutput(BaseModel):
    ticker: str
    analysis_date: str
    report_sections: Dict[str, Optional[str]]

