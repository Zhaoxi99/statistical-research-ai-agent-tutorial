"""A three-agent biomedical research demonstration using the manager pattern."""
import argparse
from dotenv import load_dotenv
from agents import Runner, ToolCallItem, trace
from specialists.coordinator_agent import QUESTION,make_coordinator

def main():
    load_dotenv()
    p=argparse.ArgumentParser();p.add_argument('--question',default=QUESTION);p.add_argument('--use-qdrant',action='store_true');args=p.parse_args()
    with trace('Lesson09 Multi-Agent Research'):
        result=Runner.run_sync(make_coordinator(args.use_qdrant),args.question,max_turns=18)
    calls=[getattr(item.raw_item,'name','unknown') for item in result.new_items if isinstance(item,ToolCallItem)]
    print('Top-level specialist calls:',calls)
    print('Final agent:',result.last_agent.name)
    print('\nREPORT\n',result.final_output)
    print('\nNote: nested specialist calls are best inspected in the SDK traces.')

if __name__=='__main__':main()
