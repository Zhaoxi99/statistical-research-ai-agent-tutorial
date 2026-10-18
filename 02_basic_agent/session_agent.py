
"""
Lesson 02: Agent with automatic session memory.
"""

from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession

load_dotenv()

agent = Agent(
    name="Statistical Research Assistant",
    instructions=(
        "You are a statistical research assistant. "
        "Answer questions precisely and remember "
        "relevant information from the conversation."
    ),
)


def main():

    # Persist messages in a local SQLite database
    session = SQLiteSession(
        "statistical_tutorial",
        "conversation_history.db"
    )

    print("Statistical Research AI Agent")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        try:
            result = Runner.run_sync(
                agent,
                question,
                session=session
            )

            print("\nAgent:", result.final_output, "\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
