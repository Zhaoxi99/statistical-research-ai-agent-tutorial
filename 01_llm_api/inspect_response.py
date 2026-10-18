
"""
Lesson 01: Inspecting an OpenAI API response.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Explain the meaning of a p-value in two sentences."
)

print("=== Generated answer ===")
print(response.output_text)

print("\n=== Response metadata ===")
print("Response ID:", response.id)
print("Model:", response.model)
print("Status:", response.status)

if response.usage is not None:
    print("\n=== Token usage ===")
    print("Input tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
