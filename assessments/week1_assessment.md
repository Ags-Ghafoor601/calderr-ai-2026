# Week 1 Assessment

## 1. Explain the difference between a language model and an agent. What capabilities does an agent add?
A language model is simply a statistical tool that predicts the next sequence of tokens given an input context. It is reactive; it only responds when prompted and has no ability to interact with the world or maintain long-term goal-oriented loops by itself. 

An AI agent builds upon a language model by adding a cognitive architecture that enables proactive behavior. The key capabilities an agent adds are:
- **Planning and Reasoning**: The ability to break down a complex goal into a sequence of steps (like the ReAct pattern).
- **Memory**: Maintaining state and conversation history across multiple turns (short-term) or using vector databases for retrieval (long-term).
- **Tool Use (Acting)**: The ability to interface with external systems, execute code, query databases, or browse the web to fetch real-time information or take actions in the real world.

## 2. What is the 'context window' and why does it matter for agentic systems?
The context window is the maximum number of tokens (words or sub-words) that an LLM can process at any one time (both input and output combined). 
For agentic systems, the context window is critical because the agent must fit its system prompt, its list of available tools, the history of previous conversation turns, retrieved documents (RAG), and its internal "scratchpad" of reasoning all into this window. If the context window is too small, the agent will "forget" earlier steps in its reasoning loop or lose access to crucial context, leading to hallucinations or loop failures.

## 3. Describe the ReAct pattern. When would you use it versus a simple chain?
The ReAct (Reasoning and Acting) pattern is a framework where an LLM alternates between generating reasoning traces ("Thought") and task-specific actions ("Action"). After taking an action (like calling a tool), it receives an "Observation" from the environment, and then reasons about what to do next based on that observation.

You would use a simple chain (like LangChain's LCEL) for deterministic, linear tasks where the sequence of operations is known in advance (e.g., Extract text -> Summarize -> Translate). You would use the ReAct pattern when the path to the solution is unknown and requires dynamic decision making, such as answering a question that might require querying a database, reading the result, and then deciding if another query is needed before providing the final answer.

## 4. What is LCEL in LangChain? Write a 5-line example of a chain using the pipe operator.
LCEL (LangChain Expression Language) is a declarative way to easily compose LangChain components (like prompts, models, and output parsers) together into a single, executable chain using the pipe (`|`) operator. It standardizes how data flows through the components, automatically supporting streaming and async operations.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

chain = prompt | model | parser
print(chain.invoke({"topic": "agents"}))
```

## 5. Explain the role of temperature in LLM sampling. When would you set it to 0?
Temperature is a hyperparameter that controls the randomness or creativity of the LLM's output during decoding. A higher temperature (e.g., 0.8 or 1.0) flattens the probability distribution of the next token, making less likely tokens more probable to be chosen, resulting in more varied and "creative" text. A lower temperature (e.g., 0.1) sharpens the distribution, favoring the most likely tokens.

You would set temperature to `0` (or very close to it) when you need deterministic, highly factual, and predictable outputs. This is especially important for Agentic systems when generating JSON for tool calls, extracting specific data from a text, or writing syntax-strict code, as any hallucinated creativity in these tasks will cause the system to crash.

## 6. Design a simple agent architecture for a customer support chatbot. What tools would it need?
**Architecture:**
1. **Input Layer**: User interface (React frontend) that sends messages to a backend API (FastAPI).
2. **State/Memory**: A session manager (backed by SQLite/Redis) to retrieve previous conversation turns.
3. **Agent Core (ReAct Loop)**: An LLM equipped with a specific system prompt instructing it to act as a polite, helpful customer support agent.
4. **Tools/Environment**: The agent has access to specific tools it can call.
5. **Output Layer**: A parser that formats the agent's final response back to the user interface.

**Required Tools:**
- `Search_Knowledge_Base`: Connects to a vector database (RAG) to look up company policies, FAQs, and troubleshooting guides.
- `Lookup_Order_Status`: Takes an Order ID and queries the company's backend database to return shipping and payment status.
- `Escalate_To_Human`: A tool that flags the ticket for manual review by a human operator and informs the user if the AI cannot resolve the issue.
