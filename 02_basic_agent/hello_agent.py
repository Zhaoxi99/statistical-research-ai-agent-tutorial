
"""
Lesson 02: Creating a basic agent with the OpenAI Agents SDK.
"""

from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

# Define the agent
agent = Agent(
    name="Statistical Research Assistant",
    instructions=(
        "You are a statistical research assistant. "
        "Help researchers understand statistical methods. "
        "Provide clear and statistically accurate explanations."
    ),
)

if __name__ == "__main__":

    question = (
        "What is the difference between "
        "a confidence interval and a credible interval?"
    )

    result = Runner.run_sync(agent, question)

    print("Question:", question)
    print("\nAgent:", result.final_output)
