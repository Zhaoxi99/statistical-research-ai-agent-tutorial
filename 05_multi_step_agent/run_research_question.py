"""One-shot demonstration of a multi-step analysis and tool call sequence."""
from agents import Runner, ToolCallItem, ToolCallOutputItem, trace
from research_agent import agent

QUESTION = (
    "Investigate the association between age and biomarker levels in the biomarker "
    "dataset, adjusting for sex. Inspect the dataset, summarise relevant variables, "
    "fit the regression, assess diagnostics, and report uncertainty and limitations."
)

if __name__ == "__main__":
    with trace("Lesson 05: End-to-end biomedical analysis"):
        result = Runner.run_sync(agent, QUESTION, max_turns=12)
    print("\n=== Tool event sequence ===")
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            print("REQUESTED:", getattr(item.raw_item, "name", "unknown"))
        elif isinstance(item, ToolCallOutputItem):
            print("OUTPUT:", str(item.output)[:280], "...")
    print("\n=== Research summary ===\n", result.final_output)
