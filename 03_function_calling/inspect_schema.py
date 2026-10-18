"""Inspect the tool metadata available to the LLM. No API request is made."""

import json
from statistical_agent import describe_numbers, process_values

for tool in (describe_numbers, process_values):
    print(f"\nNAME: {tool.name}")
    print(f"DESCRIPTION:\n{tool.description}")
    print("PARAMETER SCHEMA:")
    print(json.dumps(tool.params_json_schema, indent=2))
