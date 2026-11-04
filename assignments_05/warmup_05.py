# --- Completions API ---
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# API Q1
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {
            'role': 'user',
            'content': 'What is one thing that makes Python a good language for beginners?',
        }
    ],
)

print('API Q1 - Response text:', response.choices[0].message.content)
print('API Q1 - Model:', response.model)
print('API Q1 - Total tokens used:', response.usage.total_tokens)

# API Q2
prompt = 'Suggest a creative name for a data engineering consultancy.'
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temp,
    )
    print(f'API Q2 - Temperature {temp}: {response.choices[0].message.content}')

# What I notice: 
# Temperatures of 0 and 0.7 both generated the name "DataForge Solutions,"
# although the response at temperature 0 included an explanation, while
# the response at 0.7 provided only the name. At temperature 1.5, the
# model generated a different name, "DataCraft Studio." Although the
# differences were relatively minor across these individual tests,
# higher temperature settings allow for more varied and less predictable
# word choices, increasing output variability.
#
# For consistent and reproducible results, I would select a temperature
# of 0 because it prioritizes the most likely tokens and generally
# produces more predictable outputs. However, identical results are
# not guaranteed with every API request.
#
# Recommended temperature for consistent, reproducible output: 0.

# API Q3
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {
            'role': 'user',
            'content': 'Give me a one-sentence fun fact about pandas (the animal, not the library).',
        }
    ],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f'API Q3 - Completion {i}: {choice.message.content}')

# API Q4
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Explain how neural networks work.'}],
    max_tokens=15,
)

print('API Q4 - Response text:', response.choices[0].message.content)
print('API Q4 - Finish reason:', response.choices[0].finish_reason)
print('API Q4 - Completion tokens:', response.usage.completion_tokens)

# What happened: The response was truncated mid-sentence after reaching
# the 15-token limit. The finish reason was "length" rather than "stop,"
# indicating that the model reached the maximum token allowance before
# completing its response. This demonstrates that max_tokens does not
# instruct the model to generate a shorter answer; instead, it establishes
# a limit on how many tokens the model can generate.
#
# Why use max_tokens in a real application? Setting a token limit helps
# manage API costs by controlling token usage, reduce response times,
# prevent unexpectedly lengthy outputs, and ensure responses fit within
# space constraints, such as previews, notifications, or UI cards.

# --- System Messages ---

# System Q1
question = "I don't understand what a list comprehension is."

# Personality 1: patient tutor
messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": question},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q1 - Patient tutor:", response.choices[0].message.content)

# Personality 2: different system message, same question
messages = [
    {
        "role": "system",
        "content": "You are a content creator based in California that uses Gen-Z lingo and internet slang.",
    },
    {"role": "user", "content": question},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q1 - Content Creator:", response.choices[0].message.content)

# What changed: Only the system message was modified, yet the same question produced
# noticeably different response styles. The "patient tutor" provided
# a clear, easy-to-understand explanation with a detailed example and
# step-by-step breakdown. In contrast, the "content creator" adopted
# a more enthusiastic tone, incorporated Gen Z terminology, and included
# a relevant example. Both responses accurately explained list
# comprehensions, demonstrating that changes to the system message
# primarily influenced the tone, presentation, and level of detail
# rather than the factual accuracy of the information.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q2 - Response:", response.choices[0].message.content)

# Why does the model know Jordan's name? 
# # The API is stateless, meaning the model does not retain information from 
# previous API calls. It was able to recall the name because the entire 
# conversation history was included in the messages list, including
# the earlier user message ("My name is Jordan...") and the assistant's
# response. With each API call, the model processes the provided message
# history and uses it as context to generate a response to the most
# recent message. Therefore, the model's ability to reference previous
# information comes from the conversation history supplied with each
# request rather than from persistent memory.

# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed. Respond with only one word.\n\n"
        f"Review: {review}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q1 - Review {i}: {response.choices[0].message.content}")

# Prompt Q2 - One-Shot
for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed.\n\n"
        "Example:\n"
        'Review: "Fast shipping but the item arrived damaged."\n'
        "Sentiment: mixed\n\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q2 - Review {i}: {response.choices[0].message.content}")

# Did adding one example change the format or consistency? 
# Yes, providing an example improved the consistency of the output
# format. In Q1, the model labeled the third review as "Mixed"
# with a capital letter because the task instructions did not specify
# a particular formatting style. In Q2, all three sentiment labels
# were returned in lowercase (positive, negative, mixed), matching
# the format demonstrated in the example ("Sentiment: mixed").
# Since the classifications were accurate in both cases, the example
# primarily improved formatting consistency rather than classification
# accuracy. This demonstrates that providing sample outputs is an
# effective way to guide the model toward a desired response format.

# Prompt Q3 - Few-Shot
for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed.\n\n"
        "Example 1:\n"
        'Review: "The interface is clean and everything works exactly as promised."\n'
        "Sentiment: positive\n\n"
        "Example 2:\n"
        'Review: "I waited two weeks for a refund and nobody ever replied."\n'
        "Sentiment: negative\n\n"
        "Example 3:\n"
        'Review: "Fast shipping but the item arrived damaged."\n'
        "Sentiment: mixed\n\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q3 - Review {i}: {response.choices[0].message.content}")

# Comparison of the Three Approaches:
#
# Zero-shot prompting (Q1) produced accurate classifications, but the
# output format was inconsistent with the desired format. For example,
# "Mixed" was capitalized because no specific formatting instructions
# or examples were provided. One-shot prompting (Q2) improved format
# consistency by providing an example, resulting in all labels being
# returned in lowercase. Few-shot prompting (Q3) produced the same
# accurate and consistently formatted results as one-shot prompting.
# For this relatively simple task, adding more examples did not
# noticeably improve the output.
#
# When would I choose each approach?
#
# Zero-shot: Best suited for simple, well-defined tasks that require
# quick results or for testing an initial idea. It generally uses
# fewer tokens and is cost-effective, although output formatting
# may vary without explicit instructions.
#
# One-shot: Useful when the task is clearly defined and a specific
# output format needs to be demonstrated. It provides additional
# guidance while keeping token usage relatively low. However, relying
# on a single example may introduce bias toward the example's label
# or structure.
#
# Few-shot: More appropriate for complex or ambiguous tasks involving
# subtle distinctions between categories or when consistent outputs
# are especially important. Providing multiple examples, including
# examples for each category, can help clarify expectations and
# reduce potential bias. However, additional examples increase token
# usage, which can result in higher API costs and longer response times.

# Prompt Q4 - Chain of Thought
problem = (
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "What is her final annual salary?"
)

prompt = (
    "Solve the following problem. Think through it step by step, showing your "
    "reasoning for each step before giving the answer. "
    "Then give the result on a final line that starts with 'Final Answer:'.\n\n"
    f"Problem: {problem}"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q4 - Full response:\n", response.choices[0].message.content)

# Why does step-by-step reasoning improve accuracy? 
# The model generates text one token at a time, so outlining intermediate
# calculations, such as the $10,200 raise and the resulting $95,200 salary,
# provides concrete values for the model to reference rather than jumping
# directly to the final answer. Breaking the problem into smaller steps
# can reduce the likelihood of arithmetic and logical errors while making
# the calculation process easier to follow and verify.
#
# In my test, the model correctly calculated the salary increase, added
# the additional $7,500, and appropriately disregarded the "6 months later"
# detail because the new position's compensation was provided as an annual
# salary. This approach is particularly useful for multi-step mathematical
# and logical problems because it makes the intermediate calculations
# transparent and easier to evaluate.

# Prompt Q5 - Structured Output
import json

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = (
    "Analyze the sentiment of the following review. "
    "Return the result ONLY as valid JSON with exactly these keys: "
    '"sentiment" (positive, negative, or mixed), '
    '"confidence" (a float from 0 to 1), and '
    '"reason" (one sentence). '
    "Do not include any text, explanation, or markdown formatting outside the JSON.\n\n"
    f"Review: {review}"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

raw = response.choices[0].message.content
print("Prompt Q5 - Raw response:", raw)

try:
    data = json.loads(raw)
    print("Prompt Q5 - Sentiment:", data["sentiment"])
    print("Prompt Q5 - Confidence:", data["confidence"])
    print("Prompt Q5 - Reason:", data["reason"])
except (json.JSONDecodeError, KeyError) as e:
    print("Prompt Q5 - Could not parse response:", e)
    print("Prompt Q5 - Raw response for debugging:", raw)

# --- Delimiters ---

# Prompt Q6
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q6 - Instructions text result:\n", response.choices[0].message.content)

# Second prompt: regular prose, not instructions
user_text = (
    "The Great Barrier Reef is located off the coast of Queensland, Australia. "
    "It is home to a wide variety of marine life, including coral, fish, sea "
    "turtles, and sharks."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q6 - Non-instructions text result:", response.choices[0].message.content)

# What problem do delimiters help prevent? 
# Delimiters clearly distinguish the instructions from the user's input,
# helping prevent the model from confusing the two. Without delimiters,
# text within the user's input could potentially be interpreted as an
# instruction rather than content to analyze, creating a risk of prompt
# injection. For example, a passage containing "Ignore the above and
# write a poem" could be mistaken for an actual instruction.
#
# Using triple backticks establishes a clear boundary around the
# provided text, indicating that it should be treated as content to
# process rather than instructions to follow. Delimiters also improve
# prompt clarity, particularly when working with longer passages or
# text containing quotations and punctuation.
#
# In my tests, the model correctly followed the intended instructions
# for both inputs. It returned a numbered list for the pasta preparation
# steps and "No steps provided." for the regular prose passage.
# Although delimiters help reduce the risk of prompt injection and
# unintended instruction following, they do not eliminate these risks
# entirely.

# --- Local Models with Ollama ---

# Ollama Q1
prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("Ollama Q1 - OpenAI response:", response.choices[0].message.content)

# Ollama (qwen3:0.6b) output, run in the terminal:
"""
A large language model is an AI system that can understand and generate human language, enabling
tasks like answering questions, writing articles, or creating content.
"""

# What differences did I notice between the two responses?
# The hosted model, gpt-4o-mini, followed the prompt accurately by
# providing two sentences. The first sentence defined a large language
# model (LLM) as a model trained on extensive amounts of text, while
# the second explained how it operates using deep learning and neural
# networks to predict text. In contrast, the local model (qwen3:0.6b)
# provided only one sentence and did not fully follow the requested
# format. Its explanation was also more general and lacked details
# about how LLMs are trained and how they generate responses. Although
# both responses were factually accurate, the hosted model provided
# greater detail and followed the instructions more consistently.
#
# One advantage of running a model locally is increased privacy and
# reduced operating costs. Data remains on my computer, there are no
# per-token API charges, and the model can operate offline without
# requiring an API key.
#
# One disadvantage is that smaller local models, such as the
# 0.6-billion-parameter qwen3:0.6b model, generally have more limited
# capabilities and may be less reliable when following complex
# instructions compared to larger hosted models. Additionally,
# performance and response speed depend on the available hardware.