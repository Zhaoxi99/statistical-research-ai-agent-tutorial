"""Evaluate operational characteristics, NOT statistical correctness."""
import argparse,asyncio,csv,time,statistics
from pathlib import Path
from orchestration import execute_plan
from planning import example_plan
async def worker(step,prior):
    await asyncio.sleep(.08 if step.role=='statistics' else .12)
    return 'Synthetic successful task '+step.id
async def compare():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=5);args=parser.parse_args()
    if not 1<=args.repeats<=100:parser.error('--repeats must be 1..100')
    rows=[]
    for r in range(args.repeats):
        for label,n in [('sequential',1),('parallel',2)]:
            start=time.perf_counter()
            result,waves=await execute_plan(example_plan(),worker,max_parallel=n)
            rows.append({'repeat':r+1,'configuration':label,'seconds':round(time.perf_counter()-start,4),
              'completed_steps':sum(o.status=='ok' for o in result.values()),'waves':str(waves)})
    path=Path(__file__).with_name('orchestration_results.csv')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for label in ('sequential','parallel'):
        vals=[x['seconds'] for x in rows if x['configuration']==label]
        print(f'{label}: mean={statistics.mean(vals):.3f}s across {len(vals)} synthetic runs')
    print('Saved',path.name,'; these timings are simulated, not API performance measurements.')
asyncio.run(compare())
