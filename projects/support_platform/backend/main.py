import os
import time
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="../../.env")

app = FastAPI(title="AI Customer Support Platform API")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Logging Middleware for API requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Time: {process_time:.4f}s")
    return response

# Basic Rate Limiting (In-memory simple implementation for demonstration)
RATE_LIMIT_DURATION = 60
RATE_LIMIT_REQUESTS = 20
ip_request_counts = {}

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip not in ip_request_counts:
        ip_request_counts[client_ip] = []
        
    # Remove old requests
    ip_request_counts[client_ip] = [t for t in ip_request_counts[client_ip] if current_time - t < RATE_LIMIT_DURATION]
    
    if len(ip_request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return Response("Rate limit exceeded", status_code=429)
        
    ip_request_counts[client_ip].append(current_time)
    response = await call_next(request)
    return response


# Pydantic Models
class StartSessionResponse(BaseModel):
    session_id: str

class SendMessageRequest(BaseModel):
    session_id: str
    message: str

class SendMessageResponse(BaseModel):
    response: str

class ChatHistoryResponse(BaseModel):
    messages: list[dict]


# Langchain Setup
DB_URL = "sqlite:///./chat_history.db"

TEMPLATE = """You are a polite, helpful, and professional customer support agent.
You assist users with their inquiries, troubleshoot problems, and maintain a friendly tone.
Do not make up fake policies or promise refunds. If you do not know the answer, politely state that you can escalate the issue to a human agent."""

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=DB_URL)

def get_conversation_chain():
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise e

    prompt = ChatPromptTemplate.from_messages([
        ("system", TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    return conversation


# Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/sessions", response_model=StartSessionResponse)
def start_session():
    session_id = str(uuid.uuid4())
    logger.info(f"Started new session: {session_id}")
    return {"session_id": session_id}

@app.post("/messages", response_model=SendMessageResponse)
def send_message(req: SendMessageRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        conversation = get_conversation_chain()
        response = conversation.invoke(
            {"input": req.message},
            config={"configurable": {"session_id": req.session_id}}
        )
        return {"response": response.content}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
def get_history(session_id: str):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        role = "user" if msg.type == "human" else "agent"
        messages.append({"role": role, "content": msg.content})
    return {"messages": messages}
