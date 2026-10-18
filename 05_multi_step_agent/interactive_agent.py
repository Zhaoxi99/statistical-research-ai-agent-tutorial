"""Ask questions interactively and view the requested tool calls."""
from agents import Runner, ToolCallItem, trace
from research_agent import agent


def main():
    history = []
    print("Multi-step Statistical Research Agent (type 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            with trace("Lesson 05: Statistical research workflow"):
                result = Runner.run_sync(agent, history + [{"role": "user", "content": question}], max_turns=12)
            calls = [getattr(item.raw_item, "name", "unknown")
                     for item in result.new_items if isinstance(item, ToolCallItem)]
            print("Tool calls:", calls if calls else "None")
            print("\nAgent:", result.final_output)
            history = result.to_input_list()
        except Exception as exc:
            print(f"Run failed: {type(exc).__name__}: {exc}")
            # Do not add a failed attempt to the history.

if __name__ == "__main__":
    main()
