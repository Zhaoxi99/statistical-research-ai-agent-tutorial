"""Repeated exploratory architecture comparison. Saves JSONL for later analysis."""
import argparse,json,time,datetime
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent,Runner,ToolCallItem
from specialists.coordinator_agent import make_coordinator,QUESTION
from specialists.statistical_agent import statistics_agent
from specialists.literature_agent import make_literature_agent
from tools.agent_tools import inspect_data,describe_variable,fit_regression,check_regression_assumptions

OUT=Path(__file__).parent/'evaluation_runs.jsonl'

def single_agent():
    return Agent(name='Single Research Agent',
       instructions=('Handle numerical analysis with supplied deterministic tools and research with web search. '
                     'You must inspect data, fit requested OLS, perform diagnostics and search methodology literature. '
                     'Provide source URLs and avoid causal claims.'),
       tools=[inspect_data,describe_variable,fit_regression,check_regression_assumptions]+make_literature_agent().tools)

def main():
    load_dotenv()
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=1);p.add_argument('--output',default=str(OUT));args=p.parse_args()
    if args.repeats<1: p.error('--repeats must be positive')
    dest=Path(args.output)
    for i in range(args.repeats):
        for label,build in [('single',single_agent),('multi',make_coordinator)]:
            start=time.perf_counter(); error=None;result=None
            try: result=Runner.run_sync(build(),QUESTION,max_turns=18)
            except Exception as exc: error=f'{type(exc).__name__}: {str(exc)[:250]}'
            elapsed=time.perf_counter()-start
            calls=[];tokens=None;answer=''
            if result is not None:
                calls=[getattr(item.raw_item,'name','unknown') for item in result.new_items if isinstance(item,ToolCallItem)]
                usage=getattr(getattr(result,'context_wrapper',None),'usage',None)
                tokens=getattr(usage,'total_tokens',None)
                answer=str(result.final_output)
            rec={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'architecture':label,'repeat':i+1,'latency_seconds':round(elapsed,3),
                 'tokens_observed':tokens,'top_level_tool_calls':calls,
                 'run_success':error is None,'error':error,'final_answer':answer}
            with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(label,'run',i+1,'success',rec['run_success'],'latency',rec['latency_seconds'],'top-level calls',calls,flush=True)
    print('Saved',dest)
    print('Important: SDK execution success does not prove research correctness; review answers against a rubric.')

if __name__=='__main__':main()
