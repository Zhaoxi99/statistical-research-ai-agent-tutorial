
"""
Lesson 02: Interactive agent with conversational memory.
"""

from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

agent = Agent(
    name="Statistical Research Assistant",
    instructions=(
        "You are a statistical research assistant. "
        "Help researchers understand statistical concepts. "
        "Be clear, precise, and consistent with earlier context."
    ),
)


def main():

    history = []

    print("Statistical Research AI Agent")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        # Combine previous history with the new question
        new_input = history + [
            {"role": "user", "content": question}
        ]

        try:
            result = Runner.run_sync(agent, new_input)

            print("\nAgent:", result.final_output, "\n")

            # Update history after a successful run
            history = result.to_input_list()

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
