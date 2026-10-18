"""Manager pattern: coordinator retains responsibility for final answer."""
import argparse
from dotenv import load_dotenv
from agents import Runner
from specialists.coordinator_agent import make_coordinator,QUESTION

def main():
    load_dotenv()
    p=argparse.ArgumentParser();p.add_argument('--question',default=QUESTION);p.add_argument('--use-qdrant',action='store_true');args=p.parse_args()
    result=Runner.run_sync(make_coordinator(args.use_qdrant),args.question,max_turns=15)
    print('Final agent:',result.last_agent.name)
    print(result.final_output)

if __name__=='__main__':main()
