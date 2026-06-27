# Intelligent CLI Assistant

An AI-powered terminal assistant focused on Programming and Software Engineering. Built with Python, LangChain, Groq, and Rich.

## Features
- **Domain Focus:** Specialized in software engineering, coding, and debugging.
- **Conversation Memory:** Remembers context across multiple interactions in the same session.
- **Rich Terminal UI:** Beautifully formatted markdown output, code highlighting, and thinking spinners.
- **Commands:** 
  - `/clear`: Wipes the conversation memory to start fresh.
  - `/exit`: Closes the application.

## Setup Instructions
1. Ensure you have activated your virtual environment (e.g., `source ../../calderr-env/bin/activate`).
2. Navigate to this directory: `cd projects/cli_assistant`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure your `.env` file is present in the root `Calder Internship` directory with your `GROQ_API_KEY` defined.
5. Run the assistant:
   ```bash
   python main.py
   ```

## Example Conversations
1. **User:** Write a Python function to calculate the Fibonacci sequence.
   **Assistant:** (Provides code block and explanation)
2. **User:** Can you optimize that using memoization?
   **Assistant:** (Recalls previous context and provides memoized version)
3. **User:** What are the ingredients for pancakes?
   **Assistant:** (Politely steers conversation back to programming or refuses playfully)
