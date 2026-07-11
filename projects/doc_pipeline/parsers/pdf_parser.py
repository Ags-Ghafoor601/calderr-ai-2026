"""
PDF Parser — Extracts text from PDF files using PyMuPDF
=========================================================
"""

import pymupdf  # PyMuPDF


def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_bytes: Raw bytes of the PDF file

    Returns:
        Extracted text as a single string

    Raises:
        ValueError: If the PDF cannot be read or is empty
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {str(e)}")

    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    text_parts = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

    doc.close()

    if not text_parts:
        raise ValueError("PDF contains no extractable text (may be image-based)")

    return "\n\n".join(text_parts)
