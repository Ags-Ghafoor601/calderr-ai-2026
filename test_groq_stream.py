import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# Stream the response token by token
print("Streaming response:\n")
for chunk in chain.stream({"question": "Explain how LangChain chains work in simple terms."}):
    print(chunk, end="", flush=True)

print("\n\nDone.")