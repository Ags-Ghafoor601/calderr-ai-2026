# Prompt Engineering

Prompt engineering is the art of crafting effective inputs to get desired outputs from LLMs.

## Core Techniques

### Zero-Shot Prompting
Ask the model directly without any examples.
"Classify the sentiment of this review: 'The movie was fantastic!'"

### Few-Shot Prompting
Provide a few examples before the actual task.
This helps the model understand the expected format and reasoning pattern.

### Chain-of-Thought (CoT)
Instruct the model to think step-by-step.
"Let's think step by step" dramatically improves reasoning accuracy.

### ReAct (Reasoning + Acting)
Combine reasoning traces with actions.
The model thinks about what to do, takes an action, observes the result, and repeats.

## Advanced Techniques
- **Self-Consistency**: Generate multiple answers and take the majority vote
- **Tree of Thoughts**: Explore multiple reasoning paths in a tree structure
- **Prompt Chaining**: Break complex tasks into a series of simpler prompts
- **System Prompts**: Set the model's role, tone, and constraints

## Anti-Patterns to Avoid
- Vague or ambiguous instructions
- Missing context or constraints
- Not specifying output format
- Ignoring the model's limitations
- Prompt injection vulnerabilities