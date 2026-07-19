"""Tests for the Enterprise Document Intelligence Platform API."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------
class TestHealth:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------
class TestTenants:
    def test_create_tenant(self):
        response = client.post("/api/v1/tenants/", json={
            "name": "Test Corp",
            "description": "A test tenant",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Corp"
        assert "tenant_id" in data
        assert data["tenant_id"] == "test_corp"

    def test_list_tenants(self):
        # Create a tenant first
        client.post("/api/v1/tenants/", json={"name": "List Test"})
        response = client.get("/api/v1/tenants/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_tenant(self):
        # Create first
        client.post("/api/v1/tenants/", json={"name": "Get Test"})
        response = client.get("/api/v1/tenants/get_test")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "get_test"

    def test_get_nonexistent_tenant(self):
        response = client.get("/api/v1/tenants/nonexistent_xyz")
        assert response.status_code == 404

    def test_delete_tenant(self):
        client.post("/api/v1/tenants/", json={"name": "Delete Me"})
        response = client.delete("/api/v1/tenants/delete_me")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class TestDocuments:
    def test_list_documents_empty(self):
        client.post("/api/v1/tenants/", json={"name": "Doc Test"})
        response = client.get("/api/v1/documents/doc_test")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "doc_test"
        assert isinstance(data["documents"], list)

    def test_upload_invalid_tenant(self):
        response = client.post(
            "/api/v1/documents/nonexistent/upload",
            files={"file": ("test.txt", b"test content", "text/plain")},
        )
        assert response.status_code == 404

    def test_upload_invalid_file_type(self):
        client.post("/api/v1/tenants/", json={"name": "Upload Test"})
        response = client.post(
            "/api/v1/documents/upload_test/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class TestQuery:
    def test_query_nonexistent_tenant(self):
        response = client.post("/api/v1/query/nonexistent", json={
            "question": "What is this?",
        })
        assert response.status_code == 404

    def test_query_validation(self):
        client.post("/api/v1/tenants/", json={"name": "Query Test"})
        response = client.post("/api/v1/query/query_test", json={
            "question": "ab",  # Too short (min 3 chars)
        })
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
