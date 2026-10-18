"""Define agent tools and compare clear versus vague tool descriptions."""

from dotenv import load_dotenv
from agents import Agent, ModelSettings, function_tool
from statistical_tools import calculate_statistics

load_dotenv()


@function_tool

def describe_numbers(values: list[float]) -> dict:
    """Calculate sample size, mean, sample standard deviation, min and max.

    Use this tool for questions asking to compute numerical summary or
    descriptive statistics. Do not use it for purely conceptual questions.

    Args:
        values: At least two finite numerical observations to summarise.
    """
    print(f"TOOL EXECUTED: describe_numbers({values})")
    try:
        return calculate_statistics(values)
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


@function_tool

def process_values(values: list[float]) -> dict:
    """Process some values.

    Args:
        values: Some numbers.
    """
    print(f"TOOL EXECUTED: process_values({values})")
    try:
        return calculate_statistics(values)
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}


BASE_INSTRUCTIONS = (
    "You are a statistical research assistant. Answer questions accurately. "
    "Use an available tool when a numerical computation is needed, and base "
    "numerical results on tool outputs. Conceptual questions may not need tools. "
    "When a tool returns an error, explain it rather than inventing a result."
)


def make_agent(description: str = "clear") -> Agent:
    """Create two comparable agents differing only in tool metadata."""
    if description not in {"clear", "vague"}:
        raise ValueError("description must be 'clear' or 'vague'")
    tool = describe_numbers if description == "clear" else process_values
    return Agent(
        name="Statistical Research Assistant",
        instructions=BASE_INSTRUCTIONS,
        tools=[tool],
        model_settings=ModelSettings(tool_choice="auto"),
    )


agent = make_agent("clear")
