import sqlite3
import json

conn = sqlite3.connect("projects/doc_pipeline/documents.db")
cursor = conn.cursor()
cursor.execute("SELECT id, filename, error FROM documents WHERE status='failed'")
for row in cursor.fetchall():
    print(f"ID: {row[0]}")
    print(f"File: {row[1]}")
    print(f"Error: {row[2]}\n")
