from agents import Agent
from specialists.statistical_agent import statistics_agent
from specialists.literature_agent import make_literature_agent

QUESTION = ('Investigate the association between age and biomarker in the biomarker dataset, '
            'adjusting for sex. Inspect missingness, fit OLS, check diagnostics, '
            'and consult methodological literature about diagnostic limitations. '
            'Write a sourced research report with limitations and no causal claims.')

def make_coordinator(use_qdrant: bool = False) -> Agent:
    literature=make_literature_agent(use_qdrant=use_qdrant)
    return Agent(
        name='Research Coordinator',
        instructions=('You must delegate numerical data investigation to statistical_specialist '
                      'and methodological literature work to literature_specialist when both are requested. '
                      'Retain final ownership of the user response. Synthesize the two results, distinguish '
                      'the dataset findings from external evidence, preserve links to sources, '
                      'never invent tool results or causal conclusions. Report missing work clearly.'),
        tools=[statistics_agent.as_tool(tool_name='statistical_specialist',
                                        tool_description='Inspect dataset, fit OLS, report actual coefficients and diagnostics.'),
               literature.as_tool(tool_name='literature_specialist',
                                  tool_description='Search scientific/statistical methodology and supply sources.')],
    )
