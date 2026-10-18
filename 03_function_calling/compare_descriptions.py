"""Exploratory comparison of tool-selection behaviour under two descriptions.

Each prompt is a fresh run (no conversational history). Review outcomes; a
called tool need not have completed successfully or produced a correct answer.
"""

import csv
from agents import Runner, ToolCallItem
from statistical_agent import make_agent

QUESTIONS = [
    ("Calculate the sample mean and SD of 12, 15, 18, 20, and 25.", True),
    ("What is the average of 4, 8, 12 and 16?", True),
    ("What does sample standard deviation mean?", False),
    ("Explain the difference between a p-value and a Bayes factor.", False),
    ("Find the minimum and maximum of 6, 9, 13 and 22.", True),
]
REPEATS = 2  # Increase for a more informative experiment; costs API tokens.


def main():
    records = []
    for description in ("clear", "vague"):
        agent = make_agent(description)
        for question, expected_call in QUESTIONS:
            for repeat in range(1, REPEATS + 1):
                row = {
                    "description": description,
                    "question": question,
                    "repeat": repeat,
                    "expected_tool_use": expected_call,
                }
                try:
                    result = Runner.run_sync(agent, question)
                    calls = [
                        item.raw_item.name
                        for item in result.new_items
                        if isinstance(item, ToolCallItem)
                        and getattr(item.raw_item, "name", None)
                    ]
                    row.update(
                        tool_calls=";".join(calls),
                        called_tool=bool(calls),
                        correct_decision=bool(calls) == expected_call,
                        final_answer=str(result.final_output),
                        error="",
                    )
                except Exception as exc:
                    row.update(
                        tool_calls="", called_tool=False,
                        correct_decision=False, final_answer="",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                records.append(row)
                print(description, repeat, row["correct_decision"], question)

    with open("tool_selection_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    for description in ("clear", "vague"):
        subset = [r for r in records if r["description"] == description]
        successes = sum(r["correct_decision"] for r in subset)
        failures = sum(bool(r["error"]) for r in subset)
        print(f"{description}: decision match {successes}/{len(subset)}, run failures {failures}")
    print("Saved tool_selection_results.csv")


if __name__ == "__main__":
    main()
