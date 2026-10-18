"""Interactive data-analysis agent. Run from terminal, not during imports."""
from agents import Runner, ToolCallItem
from data_agent import agent


def main() -> None:
    history = []
    print("Data Analysis Agent — type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            result = Runner.run_sync(agent, history + [{"role": "user", "content": question}])
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}\n")
            continue
        history = result.to_input_list()
        calls = [
            getattr(item.raw_item, "name", "")
            for item in result.new_items
            if isinstance(item, ToolCallItem)
        ]
        print("Tool calls:", calls or "None")
        print("Agent:", result.final_output, "\n")


if __name__ == "__main__":
    main()
