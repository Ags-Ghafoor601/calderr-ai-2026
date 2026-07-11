import pytest
import os
from extraction.extractor import extract_document
from models import DocumentExtraction

def test_extractor_json_schema():
    with open("test_documents/meeting_notes.txt", "r") as f:
        text = f.read()
    
    # We test that extract_document does not crash and returns the correct Pydantic model
    # The tenacity retry logic will handle any transient rate limits or parsing errors
    result = extract_document(text)
    
    assert isinstance(result, DocumentExtraction)
    assert len(result.summary) > 0
    assert len(result.entities) > 0
    assert result.document_type_guess in ["meeting_notes", "memo", "general", "report"]
