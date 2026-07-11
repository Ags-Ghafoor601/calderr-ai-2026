import pytest
from fastapi.testclient import TestClient
from main import app, init_db
import asyncio

@pytest.fixture(scope="session")
def client():
    # Initialize the database before running tests
    asyncio.run(init_db())
    return TestClient(app)

def test_upload_documents(client):
    filenames = ["invoice.txt", "meeting_notes.txt", "project_proposal.txt"]
    for filename in filenames:
        with open(f"test_documents/{filename}", "rb") as f:
            response = client.post("/api/upload", files={"file": (filename, f, "text/plain")})
            
            assert response.status_code == 200, f"Failed for {filename}: {response.text}"
            data = response.json()
            
            # Verify the response structure matches our Pydantic model
            assert "extraction" in data
            assert data["status"] == "completed"
            assert "summary" in data["extraction"]
            assert isinstance(data["extraction"]["entities"], list)
