import re

def read_log_file(file_path: str) -> list[str]:
    """Read a log file and return non-empty lines with trailing newlines stripped."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [clean_text(line) for line in f if line.rstrip("\n")]

    print("\n ********************* File Read Complete *********************")
    print(f"Read {len(lines)} lines.")

    return lines

def clean_text(text: str):
    clean_text = text.rstrip("\n")
    clean_text = re.sub(r'\t\.\.\. \d+ more', '', clean_text)
    return clean_text