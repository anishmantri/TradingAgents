
import re

def escape_latex(text: str) -> str:
    """Escape all LaTeX reserved characters."""
    if not text:
        return ""
    
    # First, handle the specific case of "newline" text artifacts
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
    
    safe_text = ""
    for char in text:
        if char in replacements:
            safe_text += replacements[char]
        else:
            safe_text += char
            
    return safe_text

def _process_formatting(text: str) -> str:
    """Helper to handle **bold** and *italic* text."""
    # Escape special characters first
    escaped = escape_latex(text)
    
    # Bold: **text** -> \textbf{text}
    escaped = re.sub(r'\*\*(.*?)\*\*', r"\\textbf{\1}", escaped)
    
    # Italic: *text* -> \textit{text}
    escaped = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r"\\textit{\1}", escaped)
    
    return escaped

def _latex_format_text(text: str) -> str:
    """Parses Markdown-like text and converts it to LaTeX."""
    if not text:
        return ""
    
    # Normalize newlines
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove "newline" artifacts
    text = re.sub(r'\bnewline\b', ' ', text, flags=re.IGNORECASE)
    
    lines = text.split("\n")
    formatted_lines = []
    in_list = False
    in_code_block = False
    in_table = False
    table_rows = []
    
    for line in lines:
        stripped = line.strip()
        
        # Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                formatted_lines.append(r"\end{verbatim}")
                in_code_block = False
            else:
                if in_list:
                    formatted_lines.append(r"\end{itemize}")
                    in_list = False
                formatted_lines.append(r"\begin{verbatim}")
                in_code_block = True
            continue
            
        if in_code_block:
            formatted_lines.append(line)
            continue

        # Tables (basic support)
        if "|" in line and (line.strip().startswith("|") or line.strip().endswith("|")):
            if not in_table:
                in_table = True
                table_rows = []
            
            # Skip separator lines like |---|---|
            if set(line.strip()) <= {"|", "-", " ", ":"}:
                continue
                
            # Parse row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        elif in_table:
            # End of table
            in_table = False
            if table_rows:
                num_cols = len(table_rows[0])
                col_spec = "l" * num_cols
                formatted_lines.append(r"\begin{tabular}{" + col_spec + "}")
                formatted_lines.append(r"\toprule")
                
                # Header
                header_cells = [_process_formatting(c) for c in table_rows[0]]
                formatted_lines.append(" & ".join(rf"\textbf{{{c}}}" for c in header_cells) + r" \\")
                formatted_lines.append(r"\midrule")
                
                # Body
                for row in table_rows[1:]:
                    if len(row) != num_cols:
                        continue
                    row_cells = [_process_formatting(c) for c in row]
                    formatted_lines.append(" & ".join(row_cells) + r" \\")
                    
                formatted_lines.append(r"\bottomrule")
                formatted_lines.append(r"\end{tabular}")
                formatted_lines.append(r"\par\vspace{0.5em}")
            table_rows = []

        if not stripped:
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            formatted_lines.append(r"\par\vspace{0.5em}") 
            continue
            
        # Headers
        if stripped.startswith("###"):
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            content = stripped.lstrip("#").strip()
            formatted_lines.append(rf"\paragraph*{{{escape_latex(content)}}}")
            continue
        elif stripped.startswith("##"):
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            content = stripped.lstrip("#").strip()
            formatted_lines.append(rf"\subsubsection*{{{escape_latex(content)}}}")
            continue
            
        # List items
        is_list_item = stripped.startswith(("- ", "* ", "• "))
        if is_list_item:
            content = stripped[1:].strip()
            if not in_list:
                formatted_lines.append(r"\begin{itemize}")
                in_list = True
            
            content = _process_formatting(content)
            formatted_lines.append(rf"\item {content}")
        else:
            if in_list:
                formatted_lines.append(r"\end{itemize}")
                in_list = False
            
            content = _process_formatting(stripped)
            formatted_lines.append(content + r" \\")
            
    if in_list:
        formatted_lines.append(r"\end{itemize}")
    if in_code_block:
        formatted_lines.append(r"\end{verbatim}")
    if in_table and table_rows:
        # Flush remaining table if file ended
        num_cols = len(table_rows[0])
        col_spec = "l" * num_cols
        formatted_lines.append(r"\begin{tabular}{" + col_spec + "}")
        formatted_lines.append(r"\toprule")
        header_cells = [_process_formatting(c) for c in table_rows[0]]
        formatted_lines.append(" & ".join(rf"\textbf{{{c}}}" for c in header_cells) + r" \\")
        formatted_lines.append(r"\midrule")
        for row in table_rows[1:]:
             if len(row) == num_cols:
                row_cells = [_process_formatting(c) for c in row]
                formatted_lines.append(" & ".join(row_cells) + r" \\")
        formatted_lines.append(r"\bottomrule")
        formatted_lines.append(r"\end{tabular}")
        
    return "\n".join(formatted_lines)

sample_text = """
4. **Falsification Criteria**

   The investment thesis should be considered invalidated.

   **4.1 Profitability and Margin Progression**

   - If, over **4–6 quarters** from initiation:
"""

print("--- Input ---")
print(sample_text)
print("\n--- Output ---")
try:
    output = _latex_format_text(sample_text)
    print(output)
except Exception as e:
    print(f"Error: {e}")
