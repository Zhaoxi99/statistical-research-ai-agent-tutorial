"""Bounded async task scheduler with dependency-based waves and failure propagation."""
import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable
from planning import ResearchPlan, Step

@dataclass
class Outcome:
    status: str
    text: str

async def execute_plan(plan: ResearchPlan, worker: Callable[[Step, dict[str,Outcome]], Awaitable[str]],
                       max_parallel: int = 2) -> tuple[dict[str,Outcome], list[list[str]]]:
    if max_parallel < 1: raise ValueError('max_parallel must be >=1')
    pending={s.id:s for s in plan.steps}
    results:dict[str,Outcome]={}
    waves=[]
    sem=asyncio.Semaphore(max_parallel)
    async def run_one(step:Step):
        if any(results[dep].status!='ok' for dep in step.depends_on):
            return Outcome('skipped','A prerequisite failed')
        async with sem:
            try:
                return Outcome('ok',await worker(step,dict(results)))
            except Exception as exc:
                return Outcome('failed',f'{type(exc).__name__}: {exc}')
    while pending:
        ready=[s for s in pending.values() if all(d in results for d in s.depends_on)]
        if not ready: raise RuntimeError('No runnable steps; dependency graph invalid')
        waves.append([s.id for s in ready])
        done=await asyncio.gather(*(run_one(s) for s in ready))
        for s,result in zip(ready,done):
            results[s.id]=result
            del pending[s.id]
    return results,waves
