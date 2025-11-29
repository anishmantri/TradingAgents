import re
from pathlib import Path

def escape_latex(text: str) -> str:
    """Escape all LaTeX reserved characters."""
    if not text:
        return ""
    
    # First, handle the specific case of "newline" text artifacts
    # If the text contains literal "newline" surrounded by spaces or at start/end, remove it
    text = re.sub(r'\s*newline\s*', ' ', str(text), flags=re.IGNORECASE)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    
    # Use a regex to replace all characters in one pass
    # This avoids issues where a replacement introduces a character that gets replaced again
    # (though the current set is mostly safe, it's good practice)
    # However, for simplicity and since the replacements are standard, simple iteration is fine
    # provided we handle backslash first or carefully.
    # The dictionary approach above is safe if we iterate character by character.
    
    safe_text = ""
    for char in text:
        if char in replacements:
            safe_text += replacements[char]
        else:
            safe_text += char
            
    return safe_text

def save_latex_debug(latex: str, path: Path) -> None:
    """Dump raw LaTeX for inspection."""
    try:
        with open(path, "w") as f:
            f.write(latex)
    except Exception as e:
        print(f"Failed to save debug LaTeX to {path}: {e}")

def format_number(value, decimals: int = 2, default: str = "N/A") -> str:
    """Format a number safely for LaTeX."""
    if value is None or value == "N/A":
        return default
    try:
        if isinstance(value, str):
            # Try to clean string (remove % $ ,)
            clean_val = value.replace("%", "").replace("$", "").replace(",", "").strip()
            num = float(clean_val)
        else:
            num = float(value)
        
        return f"{num:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def format_currency(value, decimals: int = 2, symbol: str = "\\$") -> str:
    """Format a value as currency."""
    formatted = format_number(value, decimals)
    if formatted == "N/A":
        return formatted
    return f"{symbol}{formatted}"

def format_percentage(value, decimals: int = 1) -> str:
    """Format a value as percentage."""
    # Handle cases where value might already be a percentage (e.g. 0.15 vs 15)
    # This is tricky without context, but we'll assume raw numbers are what they are
    # If the string has a %, we strip it and re-add it.
    
    if isinstance(value, str) and "%" in value:
        return f"{format_number(value, decimals)}\\%"
        
    formatted = format_number(value, decimals)
    if formatted == "N/A":
        return formatted
    return f"{formatted}\\%"

def format_large_number(value, decimals: int = 1, currency: bool = False) -> str:
    """Format large numbers (Millions, Billions) for readability."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        if isinstance(value, str):
             clean_val = value.replace("%", "").replace("$", "").replace(",", "").strip()
             num = float(clean_val)
        else:
            num = float(value)
        
        suffix = ""
        if abs(num) >= 1_000_000_000:
            num /= 1_000_000_000
            suffix = "B"
        elif abs(num) >= 1_000_000:
            num /= 1_000_000
            suffix = "M"
        elif abs(num) >= 1_000:
            num /= 1_000
            suffix = "K"
            
        formatted = f"{num:,.{decimals}f}{suffix}"
        if currency:
            return f"\\${formatted}"
        return formatted
    except (ValueError, TypeError):
        return str(value)
