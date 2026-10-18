"""Chat with an agent that may call the statistical Python tool."""

from agents import Runner, ToolCallItem
from statistical_agent import agent


def main() -> None:
    history = []
    print("Statistical Research Agent. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = Runner.run_sync(
                agent, history + [{"role": "user", "content": question}]
            )
        except Exception as exc:
            print(f"Run failed: {type(exc).__name__}: {exc}\n")
            continue
        history = result.to_input_list()
        called = [
            item.raw_item.name
            for item in result.new_items
            if isinstance(item, ToolCallItem)
            and getattr(item.raw_item, "name", None)
        ]
        print("Tool calls:", called or "None")
        print("Agent:", result.final_output, "\n")


if __name__ == "__main__":
    main()
