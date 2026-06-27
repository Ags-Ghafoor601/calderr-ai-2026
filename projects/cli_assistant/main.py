import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables
load_dotenv(dotenv_path="../../.env")

# Initialize Rich console
console = Console()

# Define the system prompt
TEMPLATE = """You are a knowledgeable and helpful Intelligent CLI Assistant focusing on the domain of Programming and Software Engineering.
You provide clear, accurate, and concise answers, including well-formatted code snippets when applicable.
If the user asks about topics completely unrelated to programming, gently steer the conversation back to software development, unless they explicitly use a command to switch topics."""

def main():
    console.print(Panel.fit("[bold blue]Welcome to the Intelligent Programming CLI Assistant![/bold blue]\nType [bold red]/exit[/bold red] to quit or [bold yellow]/clear[/bold yellow] to reset memory.", title="Startup"))
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
    except Exception as e:
        console.print(f"[bold red]Error initializing Groq API. Make sure GROQ_API_KEY is in your .env file.[/bold red]\nDetails: {e}")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    session_id = "default_session"

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            
            if user_input.strip() == "":
                continue
                
            if user_input.strip().lower() == "/exit":
                console.print("[bold blue]Goodbye![/bold blue]")
                break
                
            if user_input.strip().lower() == "/clear":
                if session_id in store:
                    store[session_id].clear()
                console.print("[bold yellow]Conversation history cleared.[/bold yellow]")
                continue
                
            with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response = conversation.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
            console.print("\n[bold purple]Assistant:[/bold purple]")
            console.print(Markdown(response.content))
            
        except KeyboardInterrupt:
            console.print("\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()
