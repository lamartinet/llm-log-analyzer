def read_log_file(file_path: str) -> list[str]:
    """Read a log file and return non-empty lines with trailing newlines stripped."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.rstrip("\n")]

    print("\n ********************* File Read Complete *********************")
    print(f"Read {len(lines)} lines.")

    return lines