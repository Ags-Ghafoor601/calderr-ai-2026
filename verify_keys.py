from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

print(f"GROQ_API_KEY starts with: {groq_key[:4] if groq_key else 'NOT FOUND'}")
print(f"HF_TOKEN starts with: {hf_token[:4] if hf_token else 'NOT FOUND'}")