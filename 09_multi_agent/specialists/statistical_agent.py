from agents import Agent
from tools.agent_tools import inspect_data, describe_variable, fit_regression, check_regression_assumptions

statistics_agent = Agent(
    name='Statistical Specialist',
    instructions=('You analyse only the bundled named datasets with provided Python tools. '
                  'Inspect before modelling. Report actual numeric outputs and units. '
                  'Never invent regression results or causal interpretations. '
                  'If a tool errors, state the problem. Explain diagnostic limitations.'),
    tools=[inspect_data,describe_variable,fit_regression,check_regression_assumptions],
)
