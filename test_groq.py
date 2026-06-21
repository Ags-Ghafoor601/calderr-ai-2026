import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}")
])

# Build the chain: prompt → model → output parser
chain = prompt | llm | StrOutputParser()

# Invoke and print
response = chain.invoke({"question": "What is an AI agent and why does it matter?"})
print(response)