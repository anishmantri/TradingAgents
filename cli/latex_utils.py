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
