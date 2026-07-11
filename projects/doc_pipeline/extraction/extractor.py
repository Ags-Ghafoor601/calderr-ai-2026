"""
Extraction Agent — Multi-Tool Document Extraction using Groq + LangChain
===========================================================================
Uses structured output (with_structured_output) to extract entities,
dates, key terms, summaries, and action items from document text.
"""

import os
import sys
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Resolve .env from repo root
_repo_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_repo_root / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import DocumentExtraction


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precision document analysis engine. Your task is to extract
structured information from document text.

EXTRACTION RULES:
1. **Summary**: Write a concise 2-3 sentence summary capturing the document's main purpose and content.
2. **Key Terms**: Extract 5-15 important terms, concepts, or keywords.
3. **Entities**: Find ALL named entities. You MUST classify each entity into EXACTLY ONE of these types:
   - "person" (names of individuals)
   - "organization" (companies, institutions, agencies)
   - "location" (cities, countries, addresses)
   - "date" (specific dates or date ranges)
   - "money" (dollar amounts, budgets, prices)
   - "email" (email addresses)
   - "phone" (phone numbers)
   - "url" (websites)
   - "software" (software programs, frameworks, libraries)
   - "technology" (general tech concepts or protocols)
   - "product" (commercial products)
   - "course" (educational courses or classes)
   - "other" (use ONLY this if it doesn't fit the above, DO NOT make up new types)
4. **Dates**: Extract ALL dates and deadlines with context about what they refer to.
   Normalize to ISO format (YYYY-MM-DD) when possible.
5. **Action Items**: Identify any tasks, to-dos, action items, or deliverables mentioned.
   Assign severity (high/medium/low) based on urgency language.
6. **Document Type**: Classify as one of: contract, report, email, memo, invoice,
   proposal, meeting_notes, resume, letter, article, technical_doc, general.
7. **Language**: Identify the primary language.

Be thorough and precise. Extract everything you can find. Do NOT hallucinate
entities or dates that are not in the text.""",
    ),
    (
        "human",
        "Extract all structured information from the following document:\n\n{document_text}",
    ),
])


@retry(wait=wait_exponential(min=2, max=15), stop=stop_after_attempt(5))
def extract_document(text: str) -> DocumentExtraction:
    """
    Extract structured information from document text using Groq LLM.

    Args:
        text: The document text to analyze

    Returns:
        DocumentExtraction model with all extracted information

    Raises:
        Exception: If the LLM call fails or returns invalid data
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")

    # Truncate very long documents to fit context window and Groq free tier limits
    max_chars = 4000  # Strict constraint to prevent hitting 6000 TPM limit
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n\n[... document truncated for processing ...]"

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=GROQ_API_KEY,
        max_tokens=4096,
    )

    structured_llm = llm.with_structured_output(DocumentExtraction)
    chain = EXTRACTION_PROMPT | structured_llm

    result = chain.invoke({"document_text": truncated})

    # Add word count
    result.word_count = len(text.split())

    return result
