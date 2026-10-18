"""Stage 6: Agent decides whether and how to use web search across turns."""
import argparse
from agents import Agent,Runner,WebSearchTool
from config import MODEL
ap=argparse.ArgumentParser()
ap.add_argument('--question',default='Compare two recent scientific sources about Bayesian adaptive trial designs. Provide URLs and acknowledge limitations.')
a=ap.parse_args()
agent=Agent(name='Evidence-based statistical research assistant',model=MODEL,instructions=(
    'Answer statistical research questions carefully. Search the web when current or external evidence is needed. '
    'Prefer primary papers or authoritative institutions. Identify dates, distinguish claims from evidence, '
    'cite URLs, and explain uncertainty. Never fabricate a citation.'),tools=[WebSearchTool(search_context_size='medium')])
result=Runner.run_sync(agent,a.question)
print(result.final_output)
print('\nExecution completed; see SDK traces when tracing is enabled.')
