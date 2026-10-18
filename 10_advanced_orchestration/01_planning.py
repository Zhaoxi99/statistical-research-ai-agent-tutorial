"""Stage 1: compare a fixed plan with a plan proposed by an LLM."""
import argparse
from dotenv import load_dotenv
from planning import ResearchPlan, example_plan
load_dotenv()
ap=argparse.ArgumentParser(); ap.add_argument('--live',action='store_true'); args=ap.parse_args()
print('Validated fixed plan:\n',example_plan().model_dump_json(indent=2))
if args.live:
    from agents import Agent, Runner
    planner=Agent(name='Research Planner',model='gpt-4.1-mini',output_type=ResearchPlan,
      instructions=('Propose at most 6 bounded steps, using roles only: statistics, literature, reviewer. '
      'Dependencies must point only to earlier step IDs. Do not execute actions. '
      'Include separate analysis and literature steps for combined requests.'))
    result=Runner.run_sync(planner,'Investigate age and biomarker association adjusting for sex; review literature and limitations.')
    plan=result.final_output
    print('\nLLM-generated validated plan:\n',plan.model_dump_json(indent=2))
