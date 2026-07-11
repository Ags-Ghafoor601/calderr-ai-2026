"""
TXT Parser — Handles plain text file parsing
===============================================
"""


def parse_txt(file_bytes: bytes) -> str:
    """
    Extract text content from a plain text file.

    Args:
        file_bytes: Raw bytes of the text file

    Returns:
        Extracted text as a string

    Raises:
        ValueError: If the file is empty or cannot be decoded
    """
    # Try multiple encodings
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]:
        try:
            text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise ValueError("Could not decode text file with any supported encoding")

    text = text.strip()
    if not text:
        raise ValueError("Text file is empty")

    return text
