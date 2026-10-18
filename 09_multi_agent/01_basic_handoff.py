"""Handoff routing: selected specialist owns the final answer."""
import argparse
from dotenv import load_dotenv
from agents import Agent, Runner
from specialists.statistical_agent import statistics_agent
from specialists.literature_agent import make_literature_agent

def main():
    load_dotenv()
    p=argparse.ArgumentParser(); p.add_argument('--question',default='What is the mean age in the biomarker dataset?')
    args=p.parse_args()
    triage=Agent(name='Research Triage',
                instructions='Hand off statistical data tasks to Statistical Specialist and literature tasks to Literature Specialist. Do not answer specialist requests yourself.',
                handoffs=[statistics_agent,make_literature_agent()])
    result=Runner.run_sync(triage,args.question)
    print('Final agent:',result.last_agent.name)
    print(result.final_output)

if __name__=='__main__': main()
