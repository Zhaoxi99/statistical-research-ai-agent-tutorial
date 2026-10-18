"""Stage 5: OpenAI hosted web_search (no developer-managed corpus or embeddings)."""
import argparse
from openai import OpenAI
from config import MODEL
ap=argparse.ArgumentParser()
ap.add_argument('--question',default='Find recent methodological reviews of Bayesian adaptive clinical trial designs and cite sources.')
a=ap.parse_args()
response=OpenAI().responses.create(model=MODEL,tools=[{'type':'web_search','search_context_size':'medium'}],tool_choice='required',input=a.question)
print(response.output_text)
print('\nRaw output items (including citation metadata) can be inspected with response.model_dump().')
