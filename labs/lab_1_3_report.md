# Lab 1.3: Prompt Engineering A/B Test Report

## Overview
This report documents the findings from testing five different system prompts on a sample news article about global renewable energy capacity growth.

## Prompts Tested
1. **Prompt 1 (Zero-Shot Basic)**: "Summarize the following news article."
2. **Prompt 2 (Persona & Audience)**: "You are an expert energy sector analyst. Summarize this article for a busy executive. Keep it strictly to 2 bullet points."
3. **Prompt 3 (Chain of Thought)**: "First, identify the key statistics and regions mentioned in the article. Then, provide a concise summary that highlights the main achievements and the primary challenge."
4. **Prompt 4 (Tone Constraint)**: "Summarize the article in a highly enthusiastic and optimistic tone, emphasizing the positive growth while acknowledging the challenge at the end."
5. **Prompt 5 (Format Constraint)**: "Provide a summary in exactly one sentence using no more than 20 words."

## Evaluation Metrics

### 1. Accuracy
- **High Accuracy (Prompts 2, 3)**: Persona and Chain of Thought prompts yielded the most accurate representations of the core facts (50% growth, 510 GW, China's dominance, and the IEA financing challenge).
- **Medium Accuracy (Prompts 1, 4)**: The basic prompt was accurate but lacked focus, while the enthusiastic prompt slightly exaggerated the positive tone, occasionally downplaying the severity of the financing challenge.
- **Low Accuracy (Prompt 5)**: Due to the strict 20-word limit, the model was forced to omit key details (like specific regions or the exact nature of the challenge), reducing the overall informational accuracy.

### 2. Conciseness
- **Most Concise (Prompt 5)**: The one-sentence, 20-word limit was strictly adhered to, resulting in extreme brevity.
- **Highly Concise (Prompt 2)**: The two-bullet constraint forced a very scannable, dense summary perfect for the requested "busy executive" audience.
- **Least Concise (Prompt 3)**: Chain of Thought reasoning naturally produced longer outputs because it explicitly required identifying statistics and regions before producing the final summary.

### 3. Tone
- **Neutral / Objective (Prompts 1, 3, 5)**: These prompts yielded standard, journalistic summaries.
- **Professional / Analytical (Prompt 2)**: Adopting the "expert analyst" persona shifted the tone to be more authoritative and business-oriented.
- **Enthusiastic / Optimistic (Prompt 4)**: The constraint successfully forced words like "Incredible news!" or "Fantastic growth!", drastically changing the emotional impact of the text while still delivering the facts.

## Conclusion
Adding constraints (length, formatting, tone, and audience) provides significantly more control over the LLM's output. Chain of Thought (Prompt 3) is best for ensuring no key details are missed, while Persona (Prompt 2) is highly effective at tailoring the information density for specific readers. Extreme format constraints (Prompt 5) come at the cost of factual comprehensiveness.
