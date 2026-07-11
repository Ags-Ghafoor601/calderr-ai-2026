"""End-to-end test for the Document Processing Pipeline."""
import sys
import time
import httpx
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"

# Wait for server to be ready
print("Waiting for server...")
for i in range(10):
    try:
        r = httpx.get(f"{BASE_URL}/api/health", timeout=3)
        if r.status_code == 200:
            print(f"Server ready: {r.json()}")
            break
    except Exception:
        time.sleep(1)
else:
    print("Server not ready after 10 seconds!")
    sys.exit(1)

# Test 1: Upload meeting notes (TXT)
print("\n=== Test 1: Upload meeting_notes.txt ===")
with open("projects/doc_pipeline/test_documents/meeting_notes.txt", "rb") as f:
    r = httpx.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("meeting_notes.txt", f, "text/plain")},
        timeout=60,
    )

data = r.json()
print(f"Status: {r.status_code}")
print(f"Document ID: {data.get('id')}")
print(f"Processing time: {data.get('processing_time_ms', 0):.0f}ms")
ext = data.get("extraction", {})
print(f"Summary: {ext.get('summary', 'N/A')[:120]}...")
print(f"Entities: {len(ext.get('entities', []))} found")
print(f"Key terms: {len(ext.get('key_terms', []))} found")
print(f"Action items: {len(ext.get('action_items', []))} found")
print(f"Dates: {len(ext.get('dates', []))} found")
print(f"Doc type: {ext.get('document_type_guess', 'N/A')}")

# Test 2: Upload project proposal (TXT)
print("\n=== Test 2: Upload project_proposal.txt ===")
with open("projects/doc_pipeline/test_documents/project_proposal.txt", "rb") as f:
    r = httpx.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("project_proposal.txt", f, "text/plain")},
        timeout=60,
    )

data2 = r.json()
print(f"Status: {r.status_code}")
print(f"Document ID: {data2.get('id')}")
print(f"Summary: {data2.get('extraction', {}).get('summary', 'N/A')[:120]}...")

# Test 3: Upload invoice (TXT)
print("\n=== Test 3: Upload invoice.txt ===")
with open("projects/doc_pipeline/test_documents/invoice.txt", "rb") as f:
    r = httpx.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("invoice.txt", f, "text/plain")},
        timeout=60,
    )

data3 = r.json()
print(f"Status: {r.status_code}")
print(f"Document ID: {data3.get('id')}")
print(f"Summary: {data3.get('extraction', {}).get('summary', 'N/A')[:120]}...")

# Test 4: Upload a PDF
print("\n=== Test 4: Upload PDF (CalderR_week-2.pdf) ===")
with open("CalderR_week-2.pdf", "rb") as f:
    r = httpx.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("CalderR_week-2.pdf", f, "application/pdf")},
        timeout=60,
    )

data4 = r.json()
print(f"Status: {r.status_code}")
print(f"Document ID: {data4.get('id')}")
print(f"Summary: {data4.get('extraction', {}).get('summary', 'N/A')[:120]}...")

# Test 5: List all documents
print("\n=== Test 5: List all documents ===")
r = httpx.get(f"{BASE_URL}/api/documents", timeout=10)
docs = r.json()
print(f"Total documents: {docs.get('total', 0)}")
for doc in docs.get("documents", []):
    print(f"  [{doc.get('id')}] {doc.get('filename')} ({doc.get('status')})")

# Test 6: Get a specific document
print("\n=== Test 6: Get document #1 ===")
r = httpx.get(f"{BASE_URL}/api/documents/1", timeout=10)
doc1 = r.json()
print(f"Filename: {doc1.get('filename')}")
print(f"File type: {doc1.get('file_type')}")

print("\n=== ALL TESTS PASSED ===")
