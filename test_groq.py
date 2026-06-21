import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the .env file
load_dotenv()

# Initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Build the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}") 
])

# Create the LangChain pipeline (LCEL)
chain = prompt | llm | StrOutputParser()

# Execute the call
response = chain.invoke({"question": "What is an AI agent and why does it matter?"})
print(response)