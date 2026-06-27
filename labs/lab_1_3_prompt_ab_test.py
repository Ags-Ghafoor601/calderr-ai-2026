import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# The sample article for testing
ARTICLE = """
Global renewable energy capacity added in 2023 grew by 50% compared to 2022, reaching almost 510 gigawatts (GW), according to the International Energy Agency (IEA). 
Solar PV accounted for three-quarters of additions worldwide. The largest growth took place in China, which commissioned as much solar PV in 2023 as the entire world did in 2022, while China's wind power additions rose by 66% year-on-year. 
The increases in renewable energy capacity in Europe, the United States and Brazil also hit all-time highs. 
However, the IEA notes that lack of financing for emerging and developing economies remains a key issue, holding back the transition to clean energy globally.
"""

# The 5 different system prompts to test
PROMPTS = {
    "Prompt 1 (Zero-Shot Basic)": "Summarize the following news article.",
    "Prompt 2 (Persona & Audience)": "You are an expert energy sector analyst. Summarize this article for a busy executive. Keep it strictly to 2 bullet points.",
    "Prompt 3 (Chain of Thought)": "First, identify the key statistics and regions mentioned in the article. Then, provide a concise summary that highlights the main achievements and the primary challenge.",
    "Prompt 4 (Tone Constraint)": "Summarize the article in a highly enthusiastic and optimistic tone, emphasizing the positive growth while acknowledging the challenge at the end.",
    "Prompt 5 (Format Constraint)": "Provide a summary in exactly one sentence using no more than 20 words."
}

def run_ab_test():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    print("Starting Lab 1.3: Prompt Engineering A/B Test")
    print("-" * 50)
    
    for prompt_name, system_prompt in PROMPTS.items():
        print(f"\nEvaluating: {prompt_name}")
        print(f"System Prompt: {system_prompt}")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{article}")
        ])
        
        chain = chat_prompt | llm | StrOutputParser()
        
        start_time = time.time()
        try:
            response = chain.invoke({"article": ARTICLE})
            duration = time.time() - start_time
            
            print(f"Output ({duration:.2f}s):\n{response}")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_ab_test()
