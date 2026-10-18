
"""
Lesson 01: Calling an LLM with the OpenAI API.
"""

from dotenv import load_dotenv
from openai import OpenAI

# Load API credentials, if stored in a local .env file.
load_dotenv()

# Create an OpenAI API client.
client = OpenAI()

# Send an input to the language model.
response = client.responses.create(
    model="gpt-5-mini",
    input="Write a one-sentence bedtime story about a unicorn."
)

# Extract and display the generated text.
print(response.output_text)
