from __future__ import annotations

from datetime import datetime
from dateutil.relativedelta import relativedelta


def describe_window(days: int) -> str:
    """Return a human readable description for a lookback window."""
    if days == 7:
        return "past week"
    if days == 30:
        return "past month"
    if days % 30 == 0:
        months = days // 30
        if months == 1:
            return "past month"
        return f"past {months} months"
    if days % 7 == 0:
        weeks = days // 7
        if weeks == 1:
            return "past week"
        return f"past {weeks} weeks"
    return f"past {days} days"


def lookback_start(current_date: str, days: int) -> str:
    """Compute the start date for the lookback window."""
    curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=days)
    return start_dt.strftime("%Y-%m-%d")
