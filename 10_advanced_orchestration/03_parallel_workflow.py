"""Stage 3: concurrent specialists with a dependency barrier before review."""
import argparse,asyncio,time
from dotenv import load_dotenv
from planning import example_plan
from orchestration import execute_plan
from specialists.factory import make_specialist
load_dotenv()
ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');ap.add_argument('--max-parallel',type=int,default=2);args=ap.parse_args()
async def offline_worker(step, prior):
    await asyncio.sleep(.05)
    return f'DEMO ONLY: completed {step.objective}; prior={list(prior)}'
async def live_worker(step,prior):
    from agents import Runner
    agent=make_specialist(step.role)
    context='\n'.join(f'{k}: {v.text[:4500]}' for k,v in prior.items())
    question=step.objective+'\nUse biomarker dataset where relevant.\nPrevious results:\n'+context
    result=await Runner.run(agent,question)
    return str(result.final_output)
async def main():
    before=time.perf_counter()
    results,waves=await execute_plan(example_plan(), live_worker if args.live else offline_worker,args.max_parallel)
    print('Execution waves:',waves)
    for name,result in results.items():print(f'\n{name} [{result.status}]:\n{result.text[:3000]}')
    print('Elapsed seconds:',round(time.perf_counter()-before,3))
asyncio.run(main())
