import os
from agents import Agent, WebSearchTool
from tools.retrieval_tools import search_indexed_literature

def make_literature_agent(use_qdrant: bool = False) -> Agent:
    tools=[WebSearchTool(search_context_size='medium')]
    if use_qdrant: tools.insert(0,search_indexed_literature)
    return Agent(
        name='Literature Specialist',
        instructions=('Research statistical methodology using sourced evidence. '
                      'Search the web for up-to-date reliable sources. Prefer primary and peer-reviewed sources. '
                      'Provide working source URLs if available; never fabricate citations. '
                      'Explain uncertainty and separate methodological advice from results of our dataset.'),
        tools=tools,
    )
