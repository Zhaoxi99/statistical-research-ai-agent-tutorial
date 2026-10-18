"""End-to-end research coordinator with explicit parallel step execution and final synthesis."""
import argparse,asyncio
from dotenv import load_dotenv
from agents import Agent,Runner
from planning import example_plan
from orchestration import execute_plan
from specialists.factory import make_specialist
load_dotenv()
QUESTION=('Investigate biomarker ~ age + sex in the biomarker dataset. '
          'Discuss regression diagnostics and contextualise the limitations using scientific sources.')
async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--use-qdrant',action='store_true');args=parser.parse_args()
    async def worker(step,prior):
        specialist=make_specialist(step.role,use_qdrant=args.use_qdrant)
        history='\n'.join(f'{key} [{o.status}]: {o.text[:6000]}' for key,o in prior.items())
        result=await Runner.run(specialist,step.objective+'\nDataset: biomarker, outcome: biomarker, predictors: age and sex.\n'+history)
        return str(result.final_output)
    findings,waves=await execute_plan(example_plan(),worker,max_parallel=2)
    print('Execution waves:',waves)
    for k,v in findings.items():print(f'{k}: {v.status}')
    context='\n\n'.join(f'{name} [{o.status}]:\n{o.text}' for name,o in findings.items())
    coordinator=Agent(name='Report Coordinator',model='gpt-4.1-mini',instructions=(
      'Synthesize provided specialist outputs only. Distinguish dataset computations from sourced literature. '
      'Cite URLs only if actually given. Do not conceal failures or claim causality.'))
    report=await Runner.run(coordinator,'Question: '+QUESTION+'\nSpecialist results:\n'+context)
    print('\nFINAL REPORT:\n',report.final_output)
asyncio.run(main())
